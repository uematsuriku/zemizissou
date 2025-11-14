import matplotlib.pyplot as plt
import numpy as np
import openjij as oj

# -----------------------------
#  ランダムに都市配置 & 距離行列生成
# -----------------------------
def tsp_distance(n: int):
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    XX, YY = np.meshgrid(x, y)
    distance = np.sqrt((XX - XX.T)**2 + (YY - YY.T)**2)
    return distance, (x, y)

n = 10
distance, (x_pos, y_pos) = tsp_distance(n=n)

# -----------------------------
#  都市プロット（番号付き）
# -----------------------------
plt.plot(x_pos, y_pos, 'o')
for i, (x, y) in enumerate(zip(x_pos, y_pos)):
    plt.text(x, y, str(i), fontsize=12)
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.title("TSP City Positions")
plt.show()


# -----------------------------
#  TSP の QUBO を構築
# -----------------------------
def tsp_qubo(distance, A, B):
    n = len(distance)
    x = [[f"x[{i}][{t}]" for t in range(n)] for i in range(n)]
    qubo = {}

    # --- コスト項 ---
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in range(n):
                qubo[(x[i][t], x[j][(t + 1) % n])] = distance[i, j]

    # --- ペナルティ項: 各都市は1回だけ訪れる ---
    for i in range(n):
        for t in range(n):
            for k in range(t + 1, n):
                qubo[(x[i][t], x[i][k])] = 2 * A

    # --- ペナルティ項: 各時間に1つの都市 ---
    for t in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                qubo[(x[i][t], x[j][t])] = 2 * B

    # --- 対角項 ---
    for i in range(n):
        for t in range(n):
            qubo[(x[i][t], x[i][t])] = -(A + B)

    constant = n * (A + B)
    return qubo, constant


qubo, constant = tsp_qubo(distance, A=0.5, B=0.5)

# -----------------------------
#  OpenJij で QUBO 最適化
# -----------------------------
sampler = oj.SASampler()
response = sampler.sample_qubo(qubo, num_reads=100)

# -----------------------------
#  解のデコード
# -----------------------------
def tsp_decode_sample(sample, n):
    ones = [[int(i) for i in k[2:-1].split('][')]
            for k, v in sample.items() if v == 1]

    x_value = np.zeros((n, n))
    for indices in ones:
        x_value[tuple(indices)] = 1

    tour = np.where(x_value.T == 1)[1]
    return x_value, tour


x_value, tour = tsp_decode_sample(
    list(response.lowest().samples())[0], n)

print("巡回ルート（tour）:", tour)

# -----------------------------
#  巡回ルートの総距離計算
# -----------------------------
def compute_tour_length(tour, distance):
    total = 0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]  # 最後→最初へ戻る
        total += distance[a, b]
    return total

length = compute_tour_length(tour, distance)
print("巡回ルートの総距離:", length)


# -----------------------------
#  結果のプロット（番号付き + 経路）
# -----------------------------
plt.plot(x_pos, y_pos, "o")
for i, (x, y) in enumerate(zip(x_pos, y_pos)):
    plt.text(x, y, str(i), fontsize=12)

plt.plot(x_pos[tour], y_pos[tour], "-")
plt.plot(
    [x_pos[tour[-1]], x_pos[tour[0]]],
    [y_pos[tour[-1]], y_pos[tour[0]]],
    "-"
)
plt.title("TSP Route (OpenJij)")
plt.show()
