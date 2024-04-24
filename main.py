import pyomo.environ as pyo
import math

# Create a Pyomo ConcreteModel
model = pyo.ConcreteModel()

# Sets for network buses
model.i = pyo.Set(initialize=[1, 2, 3, 4, 5])  # Buses
model.j = pyo.Set(initialize=[1, 2, 3, 4, 5])  # Alias for buses
model.slack = pyo.Set(initialize=[1])  # Slack bus

# Scalar for base power
Sbase = 100  # Power base in MW

# Table for generator characteristics
GenD = {
    1: {'pmax': 210, 'pmin': 0, 'Qmax': 157.5, 'Qmin': -157.5, 'b': 3},
    3: {'pmax': 520, 'pmin': 0, 'Qmax': 390, 'Qmin': -390, 'b': 5},
    4: {'pmax': 200, 'pmin': 0, 'Qmax': 150, 'Qmin': -150, 'b': 2},
    5: {'pmax': 600, 'pmin': 0, 'Qmax': 450, 'Qmin': -450, 'b': 1},
}

# Table for bus demands
BD = {
    1: {'Pd': 100, 'Qd': 0},
    2: {'Pd': 300, 'Qd': 98.61},
    3: {'Pd': 300, 'Qd': 98.61},
    4: {'Pd': 400, 'Qd': 131.47},
    5: {'Pd': 500, 'Qd': 0},
}

# Table for line network
LN = {
    (1, 2): {'r': 0.00281, 'x': 0.0281, 'b': 0.00712, 'limit': 400},
    (1, 3): {'r': 0.00304, 'x': 0.0304, 'b': 0.00658, 'limit': 400},
    (1, 5): {'r': 0.00064, 'x': 0.0064, 'b': 0.03126, 'limit': 400},
    (2, 3): {'r': 0.00108, 'x': 0.0108, 'b': 0.01852, 'limit': 400},
    (3, 4): {'r': 0.00297, 'x': 0.0297, 'b': 0.00674, 'limit': 400},
    (4, 5): {'r': 0.00297, 'x': 0.0297, 'b': 0.00674, 'limit': 240},
}

# Compute bij and z for the given network
bij = {(i, j): 1 / LN[(i, j)]['x'] for i, j in LN}
z = {(i, j): math.sqrt(LN[(i, j)]['x'] ** 2 + LN[(i, j)]['r'] ** 2) for i, j in LN}
th = {
    (i, j): math.atan(LN[(i, j)]['x'] / LN[(i, j)]['r'])
    if LN[(i, j)]['r'] != 0
    else math.pi / 2
    for i, j in LN
}

# Parameter for connection between buses
cx = {(i, j): 1 if (i, j) in LN else 0 for i in model.i for j in model.j}

# Variables
model.OF = pyo.Var()  # Objective function variable
model.Pij = pyo.Var(model.i, model.j, domain=pyo.Reals)  # Power flow between buses
model.Qij = pyo.Var(model.i, model.j, domain=pyo.Reals)  # Reactive power flow
model.Pg = pyo.Var(model.i, domain=pyo.Reals)  # Generator real power output
model.Qg = pyo.Var(model.i, domain=pyo.Reals)  # Generator reactive power output
model.Va = pyo.Var(model.i, domain=pyo.Reals)  # Bus voltage angle
model.V = pyo.Var(model.i, domain=pyo.Reals)  # Bus voltage magnitude

# Set bounds for generator real and reactive power
for i in GenD:
    model.Pg[i].setlb(GenD[i]['pmin'] / Sbase)
    model.Pg[i].setub(GenD[i]['pmax'] / Sbase)
    model.Qg[i].setlb(GenD[i]['Qmin'] / Sbase)
    model.Qg[i].setub(GenD[i]['Qmax'] / Sbase)

# Set bounds for voltage angles and magnitudes
model.Va.setub(math.pi / 2)  # Maximum voltage angle
model.Va.setlb(-math.pi / 2)  # Minimum voltage angle

# Set bounds for power flows
for (i, j) in LN.keys():
    limit = LN[(i, j)]['limit']
    model.Pij[i, j].setub(limit / Sbase)
    model.Pij[i, j].setlb(-limit / Sbase)

# Constraints
def eq1_rule(model, i, j):
    if cx.get((i, j), 0) == 1:
        return model.Pij[i, j] == (model.V[i] ** 2 * pyo.cos(th[(i, j)]) - model.V[i] * model.V[j] * pyo.cos(model.Va[i] - model.Va[j] + th[(i, j)])) / z[(i, j)]
    return pyo.Constraint.Skip

model.eq1 = pyo.Constraint(model.i, model.j, rule=eq1_rule)

def eq2_rule(model, i, j):
    if cx.get((i, j), 0) == 1:
        return model.Qij[i, j] == (model.V[i] ** 2 * pyo.sin(th[(i, j)]) - model.V[i] * model.V[j] * pyo.sin(model.Va[i] - model.Va[j] + th[(i, j)])) / z[(i, j)]
    return pyo.Constraint.Skip

model.eq2 = pyo.Constraint(model.i, model.j, rule=eq2_rule)

# Constraint to ensure generation meets demand
def eq3_rule(model, i):
    if i in BD:
        return model.Pg[i] - BD[i]['Pd'] / Sbase == sum(model.Pij[j, i] for j in model.j if cx.get((j, i), 0) == 1)
    return pyo.Constraint.Skip

model.eq3 = pyo.Constraint(model.i, rule=eq3_rule)

# Constraint to ensure reactive power meets demand
def eq4_rule(model, i):
    if i in BD:
        return model.Qg[i] - BD[i]['Qd'] / Sbase == sum(model.Qij[j, i] for j in model.j if cx.get((j, i), 0) == 1)
    return pyo.Constraint.Skip

model.eq4 = pyo.Constraint(model.i, rule=eq4_rule)

# Objective function - minimize cost
def objective_rule(model):
    cost = sum(
        model.Pg[i] * GenD[i]['b'] * Sbase +
        model.Pg[i] * model.Pg[i] * GenD[i]['b'] * Sbase * Sbase +
        GenD[i]['b']
        for i in GenD
    )
    return cost

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Set initial conditions for slack bus
model.V[1] = 1.0  # Voltage magnitude at slack bus
model.Va[1] = 0  # Voltage angle at slack bus

# Create and solve the model
solver = pyo.SolverFactory('ipopt')  # You can use other solvers like 'ipopt' or 'glpk'
results = solver.solve(model)

# Check if the model solved successfully
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Model solved successfully!")
else:
    print("Model could not be solved. Termination condition:", results.solver.termination_condition)

# Display the optimal generator outputs
for i in model.i:
    if model.Pg[i].value is not None:
        print(f"Generator {i}: Pg = {model.Pg[i].value * Sbase}")

# # Display the optimal bus voltages
# for i in model.i:
#     if model.V[i].value is not None:
#         print(f"Bus {i}: V

# Display the optimal bus voltages
for i in model.i:
    if model.Va[i].value is not None:
        print(f"Bus {i}: Va = {model.Va[i].value}")


# Display the optimal active power (Pg)
print("Active Power Output (Pg) in MW:")
for i in model.i:
    if model.Pg[i].value is not None:
        print(f"Generator {i}: Pg = {model.Pg[i].value * Sbase} MW")

# Display the optimal reactive power (Qg)
print("Reactive Power Output (Qg) in MVar:")
for i in model.i:
    if model.Qg[i].value is not None:
        print(f"Generator {i}: Qg = {model.Qg[i].value * Sbase} MVar")


