import numpy as np
import matplotlib.pyplot as plt       

matrixStr = ""

def initABCDViaRandom(N, fromNumber, to):
	B = np.random.randint(fromNumber, to, size=(N, N))
	C = np.random.randint(fromNumber, to, size=(N, N))
	D = np.random.randint(fromNumber, to, size=(N, N))
	E = np.random.randint(fromNumber, to, size=(N, N))

	return B, C, D, E

with open("defaultMatrix.txt") as matrixFile:
	matrixStr = [line.strip() for line in matrixFile if line.strip()]

def convertToIntMatrixElements(matrix):
	rows = len(matrix)
	cols = len(matrix[0]) if rows > 0 else 0

	newMatrix = [[0 for _ in range(cols)] for _ in range(rows)]

	try:
		for i, row in enumerate(matrix):
			for j, col in enumerate(row): 
				newMatrix[i][j] = int(col)
	except Exception as err:
		print("Error parsing from string to int", err)

	return newMatrix

def create_matrix(B, C, D, E):
    A = np.block([[B, E], [C, D]])
    print("Матрица A:")
    print(A)
    return A

def create_matrix_f(A, k, E):
    n = A.shape[0] // 2
    F = A.copy()
    print("Матрица F (копия A):")
    print(F)

    count_e_even = np.sum(E[:, 1::2] > k)
    print(f"Количество чисел в E, больших {k} в четных столбцах: {count_e_even}")

    # Вычисление произведения чисел в нечетных строках E
    product_e_odd = 1 
    for i in range(1, n, 2):  # Итерируемся по нечетным строкам
        for j in range(n):
            product_e_odd *= E[i, j]  # Умножаем произведение на каждое число в строке
    print(f"Произведение чисел в нечетных строках E: {product_e_odd}")

    if count_e_even > product_e_odd:
        print("Обмен местами С и Е симметрично.")
        F[:n, n:] = A[n:, :n]
        F[n:, :n] = A[:n, n:]
    else:
        print("Обмен местами С и В несимметрично.")
        F[:n, :n] = A[n:, :n]

    print("Матрица F после обмена подматриц:")
    print(F)
    return F

def lower_triangular_matrix(A):
    n = A.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            G[i, j] = A[i, j]
    return G

def showHeatMapGraph(matrix):
	plt.figure(figsize=(8, 6))
	plt.subplot(1, 3, 1)
	plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
	plt.colorbar(label='Значение')
	plt.title('Тепловая карта')
	plt.tight_layout()
	plt.show()

def showLineCartGraph(matrix):
	plt.subplot(1, 3, 3)
	means = np.mean(matrix, axis=1)
	plt.plot(means, marker='o', linestyle='-', color='green')
	plt.xlabel('Номер строки')
	plt.ylabel('Среднее значение')
	plt.title('Средние значения по строкам')
	plt.tight_layout()
	plt.show()

def showHistogramGraph(matrix):
	plt.subplot(1, 3, 2)
	plt.hist(matrix.flatten(), bins=10, color='skyblue', edgecolor='black')
	plt.xlabel('Значение')
	plt.ylabel('Частота')
	plt.title('Гистограмма')
	plt.tight_layout()
	plt.show()

def showGraphMatplotlib(matrix):
	showHistogramGraph(matrix)
	showLineCartGraph(matrix)
	showHeatMapGraph(matrix)

def main():
	isRandomValuesForMatrix = True

	firstLine = matrixStr.pop(0).split(' ')
	K = int(firstLine[0])
	N = int(firstLine[1])
	B = C = D = E = None

	dataFromMatrixStr = [row.split() for row in matrixStr]

	matrixA = convertToIntMatrixElements(dataFromMatrixStr)

	if isRandomValuesForMatrix:
		B, C, D, E = initABCDViaRandom(N, -10, 11)
	else:
		B = np.array(matrixA[:N])
		C = np.array(matrixA[N:2 * N])
		D = np.array(matrixA[N * 2: 3 * N])
		E = np.array(matrixA[N * 3: 4 * N])
	
	A = create_matrix(B, C, D, E)
	F = create_matrix_f(A, K, E)

	det_A = np.linalg.det(A)
	print(f"Определитель матрицы A: {det_A}")

	sum_diag_F = np.sum(np.diag(F))
	print(f"Сумма диагональных элементов матрицы F: {sum_diag_F}")

	if det_A > sum_diag_F:
	    print("Вычисление выражения: A*AT - K * F-1")
	    A_T = A.transpose()
	    result = np.dot(A, A_T)
	    print("A * AT:")
	    print(result)

	    F_inv = np.linalg.inv(F)
	    print("Обратная матрица F:")
	    print(F_inv)

	    K_F_inv = K * F_inv
	    print("K * F-1:")
	    print(K_F_inv)

	    result = result - K_F_inv
	    print("A * AT - K * F-1:")
	    print(result)
	else:
	    print("Вычисление выражения: (A-1 + G - FТ)*K")

	    A_inv = np.linalg.inv(A)
	    print("Обратная матрица A:")
	    print(A_inv)

	    G = lower_triangular_matrix(A)
	    print("Нижняя треугольная матрица G:")
	    print(G)

	    F_T = F.transpose()
	    print("Транспонированная матрица F:")
	    print(F_T)

	    result = A_inv + G - F_T
	    print("(A-1 + G - FТ):")
	    print(result)

	    result = result * K
	    print("(A-1 + G - FТ)*K:")
	    print(result)

	showGraphMatplotlib(F)

if __name__ == "__main__":
	main()