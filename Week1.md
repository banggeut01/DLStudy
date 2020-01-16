# Week1

## 1. 헬로 파이썬

* 아나콘다 배포판 설치

  * [링크](https://www.anaconda.com/distribution/)

  * python 3 이상 설치

    

* 쓰이는 외부 라이브러리

  * 넘파이

    ```python
    >>> import numpy as np
    ```

* 넘파이 배열 생성

  ```python
  >>> x = np.array([1, 2, 3])
  >>> print(x)
  [1 2 3]
  >>> type(x)
  <class 'numpy.ndarray'>
  ```

* 넘파이 산술 연산

  ```python
  >>> x = np.array([1, 2, 3])
  >>> y = np.array([4.0, 5.0])
  >>> x + y # ValueError
  ```

  ```python
  >>> y = np.array([4.0, 5.0, 6.0])
  >>> x + y
  array([5., 7., 9.])
  ```

  ```python
  >>> y = np.array([4.0, 5.0, 6])
  >>> x + y
  array([5., 7., 9.])
  ```

  int형 자동 float 형변환

* **matplotlib**

  * pyplot 모듈 사용

  ```python
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  
  >>> x = np.arange(0, 6, 0.1) # 0~6까지 0.1간격
  >>> y = np.sin(x)
  >>> plt.plot(x, y) # 그래프를 그림
  [<matplotlib.lines.Line2D object at 0x00000199B6E4FE48>]
  >>> plt.show()
  ```

  * 부가 기능

  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  
  x = np.arange(0, 6, 0.1) 
  y1 = np.sin(x)
  y2 = np.cos(x)
  
  plt.plot(x, y1, label="sin") # 라벨
  plt.plot(x, y2, linestyle = "--", label="cos") 
  
  plt.xlabel("x") # x축 이름
  plt.ylabel("y")   
  plt.title('sin & cos') # 그래프 제목
  plt.legend() # label???? 자동으로?
  plt.show()
  ```

  * plt.plot 한 뒤 plt.show() 하면 plot 했던 것 사라짐

## 2. 퍼셉트론

### 1. 퍼셉트론 구현하기

* AND 함수

  ```python
  
  ```

## 3. 신경망

### 1. 활성화 함수

* 계단 함수

  ```python
  >>> import numpy as np
  >>> def f(x):
  ...   if x > 0: return 1
  ...   else: return 0
  >>> f(3)
  1
  ```

  f(3)은 가능하지만 f(np.array([1, 2]))는 불가능

  아래처럼 처리해준다.

  ```python
  >>> def f(x):
  ...   y = x > 0
  ...   return y.astype(np.int)
  >>> f(np.array([1, 2]))
  array([1, 1])
  ```

* 계단 함수 그래프

  ```python
  import numpy as np
  import matplotlib.pylab as plt
  
  def step_f(x):
      return np.array(x > 0, dtype=np.int)
  
  X = np.arange(-5.0, 5.0, 0.1)
  Y = step_f(X)
  plt.plot(X, Y)
  plt.ylim(-0.1, 1.1)  # y축의 범위 지정
  plt.show()
  ```

* 시그모이드 함수

  ```python
  >>> def sigmoid(x):
  ...   return 1 / (1 + np.exp(-x))
  >>> x = np.array([-1.0, 1.0, 2.])
  >>> sigmoid(x)
  array([0.26894142, 0.73105858, 0.88079708])
  ```

* 시그모이드 함수 그래프

  ```python
  >>> x = np.arange(-5.0, 5.0, 0.1)
  >>> y = sigmoid(x)
  >>> plt.plot(x, y)
  [<matplotlib.lines.Line2D object at 0x000001BBDE8E23C8>]
  >>> plt.ylim(-0.1, 1.1) # y limit 범위지정
  (-0.1, 1.1)
  >>> plt.show()
  ```

* ReLU 함수

  ```python
  def relu(x):
      return np.maximum(0, x)
  ```

### 2. 다차원 배열 계산

* 다차원 배열

```python
>>> A = np.array([1, 2, 3, 4])
>>> A.ndim # 차원 수
1
>>> A.shape # 튜플을 반환한다. 배열의 형상 4*3*2는 (4,3,2)
(4,)
>>> A.shape[0]
4

>>> B = np.array([[1, 2],[3, 4], [5, 6]])
>>> np.ndim(B)
2
>>> B.shape
(3, 2)
>>> B.shape[1]
2
```

* 행렬 곱 `np.dot()`

  ```python
  >>> A = np.array([[1, 0], [1, 0]])
  >>> B = np.array([[1, 2],[3, 4]])
  >>> np.dot(A, B)
  array([[1, 2],
         [1, 2]])
  ```

  * 대응하는 차원의 원소 수를 일치시켜야 한다.

* 신경망에서 행렬 곱

  * **X, Y, W 형상 주의**

  

### 3. 신경망 구현

```
y1 = w11*x1 + w21*x2 + b1

y2 = w12*x1 + x22*x2 + b2

y2 = w13*x1 + x23*x2 + b2
```

```python
>>> X = np.array([1.0, 0.5])
>>> W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
>>> B1 = np.array([0.1, 0.2, 0.3])
>>> print(X.shape)
(2,)
>>> print(W1.shape)
(2, 3)
>>> print(B1.shape)
(3,)

>>> A1 = np.dot(X, W1) + B1
>>> A1
array([0.3, 0.7, 1.1])

>>> Z1 = sigmoid(A1)
>>> Z1
array([0.57444252, 0.66818777, 0.75026011])
```

A 은닉층

출력층 구현

```python
>>> def identity_function(x):
...   return x

>>> W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
>>> B2 = np.array([0.1, 0.2])

>>> A2 = np.dot(Z1, W2) + B2
>>> Y = identity_function(A2) # 혹은 Y = A2
>>> Y
array([0.51615984, 1.21402696])
```

### 4. 출력층 설계

> 신경망은 분류, 회귀에 모두 이용할 수 있다. 기계학습 문제는 분류와 회귀로 나뉜다. 분류, 회귀 어떤 문제느냐에 따라 출력층에서 사용하는 활성화 함수가 달라진다. 일반적으로 회귀엔 항등 함수, 분류에는 소프트맥스 함수를 사용한다.

* 항등 함수

  * 입력을 그대로 출력한다.

* 소프트맥스 함수

  ```python
  >>> a = np.array([0.3, 2.9, 4.0])
  
  >>> exp_a = np.exp(a) # 지수 함수
  >>> exp_a
  array([ 1.34985881, 18.17414537, 54.59815003])
  
  >>> sum_exp_a = np.sum(exp_a) # 지수 함수의 합
  >>> print(sum_exp_a)
  74.1221542101633
  
  >>> y = exp_a / sum_exp_a
  >>> y
  array([0.01821127, 0.24519181, 0.73659691])
  ```

  ```python
  >>> softmax(np.array([0.3, 2.9, 4.0]))
  array([0.01821127, 0.24519181, 0.73659691])
  ```

  * **소프트맥수 함수 주의 => 오버플로우**

    * `logC` = `C'`

    * 오버플로우 막을 목적으로 입력 신호 중 최댓값 이용하는게 일반적이다.

      ```python
      >>> a = np.array([1010, 1000, 990])
      >>> np.exp(a) / np.sum(np.exp(a))
      __main__:1: RuntimeWarning: invalid value encountered in true_divide
      array([nan, nan, nan]) # 제대로 계산되지 않음
      
      >>> c = np.max(a) # 1010 최대값
      >>> a - c
      array([  0, -10, -20])
      
      >>> np.exp(a - c) / np.sum(np.exp(a - c))
      array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
      ```

  ```python
  # 소프트맥스 함수
  def softmax(a):
      c = np.max(a)
      exp_a = np.exp(a - c)
      sum_exp_a = np.sum(exp_a)
      y = exp_a / sum_exp_a
      return y
  ```

  * 소프트맥스 함수 특징
    * 소프트맥스 함수의 출력은 0에서 0.1사이의 실수
    * **주의** 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않는다. `y=exp(x)` 단조 증가 함수이기 때문이다.
    * 단조 증가 함수 : 원소 a, b가 a<=b일 때, f(a) <= f(b)
  * 신경망 학습시 출력층에서 소프트맥스 함수 사용
  * 추론 단계에서 출력층의 소프트맥스 함수 생략하는 것이 일반적이다.

### 5. 손글씨 숫자 인식

> 추론 과정을 신경망의 **순전파**라고 한다.
>
> 기게학습과 마찬가지로 신경망도 두 단계로 문제를 해결한다.
>
> 훈련 -> 추론(분류)

* MNIST 데이터셋
  * 0 ~ 9 숫자 이미지

## 4. 신경망 학습

### 1. 손실함수

* 평균 제곱 오차

  ```python
  def mean_squared_error(y, t):
      return 0.5 * np.sum((y - t) ** 2)
  ```

* 교차 엔트로피 오차

  ```python
  def cross_entropy_error(y, t):
      delta = 1e-7 # 아주 작은 값 delta
      return -np.sum(t * np.log(y + delta))
  	# np.log()에 0을 입력하면 마이너스 무한대가 되어 더 이상 진행할 수 없다.
  ```

  아주 작은 값을 더해 0이 되지 않도록 한다.

* 신경망 학습시 정확도를 지표로 삼으면 안된다. 매개변수의 미분이 대부분 장소에서 0이 되기 때문이다. 그래서 손실함수를 사용한다.

### 2. 수치 미분

* 미분

  ```python
  # 나쁜 구현 예
  >>> def numerical_diff(f, x):
  ...   h = 10e-50
  ...   return (f(x + h) - f(x)) / h
  # 반올림 오차 발생
  >>> np.float32(1e-50)
  0.0
  
  # 1e-4 정도 값을 사용하면 좋은 결과를 얻는다고 알려짐.
  # 중심 차분, 중앙 차분
  >>> def numerical_diff(f, x):
  ...   h = 1e-4
  ...   return (f(x + h) - f(x - h)) / (2 * h)
  ```

* 수치 미분 예제

  * `y=0.01x^2 + 0.1x`

    ```python
    >>> def function_1(x):
    ...   return 0.01*x**2 + 0.1*x
    ```

    ```python
    # plot
    >>> x = np.arange(0.0, 20.0, 0.1)
    >>> y = function_1(x)
    >>> plt.plot(x, y)
    [<matplotlib.lines.Line2D object at 0x000001BBDFAB32C8>]
    >>> plt.show()
    ```

    ```python
    # x=5, 10일 때 미분 계산
    >>> numerical_diff(function_1, 5)
    0.1999999999990898
    >>> numerical_diff(function_1, 10)
    0.2999999999986347
    ```

* 편미분

  * `f(x0, x1) = x0^2, x1^2`

    ```python
    >>> def function_2(x):
    ...   return x[0] ** 2 + x[1] ** 2
    ```

    `x0=3, x1=4`일 때 x0에 대한 편미분

    ```python
    >>> def function_tmp1(x0):
    ...   return x0*x0 + 4.0**2.0
    
    >>> numerical_diff(function_tmp1, 3.0)
    6.00000000000378
    ```

    편미분은 변수가 하나인 미분과 마찬가지로 특정 장소의 기울기를 구한다.

* 기울기

  ```python
  >>> def numerical_gradient(f, x):
  ...   h = 1e-4
  ...   grad = np.zeros_like(x) # x와 형상 같은 배열 생성
  
  ...   for idx in range(x.size):
  ...     tmp_val = x[idx]
  
  		# f(x + h)
  ...     x[idx] = tmp_val + h
  ...     fxh1 = f(x)
  		# f(x - h)
  ...     x[idx] = tmp_val - h
  ...     fxh2 = f(x)
  
  ...     grad[idx] = (fxh1 - fxh2) / (2*h)
  ...     x[idx] = tmp_val # 값 복원
  ...   return grad
  ...
  >>> numerical_gradient(function_2, np.array([3.0, 4.0]))
  array([6., 8.])
  >>> numerical_gradient(function_2, np.array([0.0, 2.0]))
  array([0., 4.])
  >>> numerical_gradient(function_2, np.array([3.0, 0.0]))
  array([6., 0.])
  ```

