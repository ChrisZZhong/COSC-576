from matplotlib import pyplot as plt

X = ["5", "4", "3", "2", "1"]
train = [0.976, 0.985, 0.983, 0.982, 0.982]
val = [0.746, 0.715, 0.71, 0.729, 0.680]

fig, ax = plt.subplots()
ax.plot(X, train, label='Train accuracy')
ax.plot(X, val, label='Validation accuracy')
ax.set_xlabel('freeze except last X layers')
ax.set_ylabel('accuracy')
ax.set_title('freeze except last X layers - accuracy')
ax.legend()
plt.show()