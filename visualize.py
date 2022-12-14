from matplotlib import pyplot as plt

f = open("./res.txt", "r", encoding="utf-8")
text = f.read()
times = text.split("freeze first")[1:]
# epoch = 0
epochs = [i for i in range(10)]
for epoch in epochs:
    print(epoch)
    X = [i for i in range(0, 19)]
    train = []
    val = []
    for i in X:
        ep = f"epoch: {epoch}"
        eps = times[i].split(ep)
        train_ = eps[1].split("train_acc: ")[1][:5]
        train.append(float(train_))
        val_ = eps[1].split("val_acc: ")[1][:5]
        val.append(float(val_))

    # eps[1].split(" ")[-1][:5]

    # print(train)
    print(max(val))
    #
    # X = [i for i in range(0, 19)]

    # train = [0.972, 0.964, 0.969, 0.962, 0.966, 0.971, 0.972, 0.975, 0.973, 0.976, 0.980, 0.982, 0.982, 0.985,
    #          0.987, 0.992, 0.981, 0.950, 0.842]
    # val = [0.934, 0.928, 0.923, 0.916, 0.938, 0.943, 0.920, 0.912, 0.920, 0.905, 0.935, 0.922, 0.909, 0.916,
    #        0.949, 0.934, 0.900, 0.901, 0.843]

    # train = [0.972, 0.964, 0.969, 0.962, 0.966, 0.971, 0.972, 0.975, 0.973, 0.976, 0.980, 0.982, 0.982, 0.985,
    #          0.987, 0.992, 0.981, 0.950, 0.842]
    # val = [0.934, 0.928, 0.923, 0.916, 0.938, 0.943, 0.920, 0.912, 0.920, 0.905, 0.935, 0.922, 0.909, 0.916,
    #        0.949, 0.934, 0.900, 0.901, 0.843]

    fig, ax = plt.subplots()
    ax.plot(X, train, label='Train accuracy')
    ax.plot(X, val, label='Validation accuracy')
    ax.set_xlabel('freeze first X layers')
    ax.set_ylabel('accuracy')
    ax.set_title('freeze first X layers - accuracy')
    ax.legend()
    plt.show()

