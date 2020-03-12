import numpy as np

q = [0.2, 0.65, 1, 0.65, 0.2,
     0.15, 0.25, 0.5, 0.25, 0.15,
     0, 0, 0, 0, 0]
q = np.array(q)
action = []
for i in np.arange(1, -0.1, -0.5):
    for j in np.arange(-1, 1.1, 0.5):
        action.append(np.array([i, j]))
action = np.array(action)

knot_count = len(q)

optimal_action = np.array([1, 0])
Q_new = 0.9

c = 0.9
epsilon = 0.01

num = 0
den = 0
deriv_q = []
deriv_u0 = []
deriv_u1 = []

for it in range(0, knot_count):
    print(optimal_action)
    print(action[it])
    weight = np.linalg.norm(optimal_action - action[it]) + c * (q.max() - q[it] + epsilon)
    den = den + (1.0 / weight)
    num = num + (q[it] / weight)
    deriv_q.append((den * (weight + q[it] * c) - num * c) / pow((weight * den), 2))
    deriv_u0.append(((num - den * q[it]) * 2 * (action[it][0] - optimal_action[0])) / (pow(weight * den, 2)))
    deriv_u1.append(((num - den * q[it]) * 2 * (action[it][1] - optimal_action[1])) / (pow(weight * den, 2)))

Q_dash = num / den
error = Q_new - Q_dash

for it in range(0, knot_count):
    q[it] = q[it] + error * deriv_q[it]
    action[it][0] = action[it][0] + error * deriv_u0[it]
    action[it][1] = action[it][1] + error * deriv_u1[it]

y = []
for it in range(0, knot_count):
    y.append(action[it][0] * 2 - 1)
    y.append(action[it][1] * 2 - 1)
    y.append(q[it] * 2 - 1)

print(y)
for i in range(knot_count):
    print(action[i], q[i])
