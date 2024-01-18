
class Calculator:
    def __init__(self):
        # 생성자에서 스택을 초기화하고, 괄호 확인을 위한 스택을 추가로 생성함.
        self.stack = []
        # 스택이 비었음을 확인함.
        self.parentheses_stack = []

    def __del__(self):
        print(f"Calculator 인스턴스가 소멸되었습니다.")
    # 객체가 삭제될때 호출되는 소멸자

    def is_empty(self, stack):
        return len(stack) == 0
    # stack이 비어있음을 확인하는 함수
     
    def is_operator(self, char):
        return char in {'+', '-', '*', '/'}
    # char이 연산자인지 확인하는 함수

    def calculate(self, operator, operand2, operand1):
        if operator == '+':
            return operand1 + operand2
        elif operator == '-':
            return operand1 - operand2
        elif operator == '*':
            return operand1 * operand2
        elif operator == '/':
            if operand2 == 0:
                raise ValueError("0으로 나눌 수 없습니다.")
            return operand1 / operand2
        # 연산자인 operator와 피연산자 operand1,2를 받아 사칙연산을 실행하는 함수.
        # /에서는 분모에 0이 들어갈수 없기에 예외처리

    def has_higher_precedence(self, op1, op2):
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        # 연산자들에 우선순위를 부여함.
        return precedence[op1] >= precedence[op2]
        # 연산자의 우선순위를 설정하는 메서드

    
    def has_matching_parentheses(self, expression):
        self.parentheses_stack = []
        # self.parentheses_stack 비어있음
        for char in expression:
            #expression에서 하나씩 빼서 char에 대입시킴
            if char == '(':
                self.parentheses_stack.append(char)
            elif char == ')':
                if not self.is_empty(self.parentheses_stack):
                    self.parentheses_stack.pop()
                else:
                    return False
        return self.is_empty(self.parentheses_stack)
    # 괄호의 짝이 맞는지 확인하는 함수
    '''
    여는 괄호'('가 나왔을때는 self.parentheses_stack에 넣고, ')'이 나왔을때는 self.parentheses_stack이 비어있지
    않다면 마지막 요소를 제거하고 그 이외에는 False를 반환하여 빈 stack에서 popping 하지 않도록함.
    '''

    def evaluate_expression(self, expression):
        if not self.has_matching_parentheses(expression):
            raise ValueError("괄호의 쌍이 맞지 않습니다.")
    # 괄호의 쌍이 맞는지 확인하는 함수. 식평가에서 괄호 쌍이 맞지 않으면 raise를 통해 예외처리로 보냄. 
        for char in expression:
            if char.isdigit():
                # .isdigit() 문자열이 모두 숫자인지 확인하는 메서드
                self.stack.append(float(char))
                # float는 소수점 이하의 실수부분을 표현 할때 사용되는 부동소수점 숫자를 나타내는 데이터 타입

            elif char == '(':
                self.stack.append(char)
            elif char == ')':
                while not self.is_empty(self.stack) and self.stack[-1] != '(':
                    operator = self.stack.pop()
                    operand2 = self.stack.pop()
                    operand1 = self.stack.pop()
                    result = self.calculate(operator, operand2, operand1)
                    self.stack.append(result)
                self.stack.pop()  # '('을 pop하는 문장
                '''
                '('를 만나면 stack에 넣고, ')'을 만났을때 stack이 비어있지 않으며 마지막 요소가 '('이 아니라면 
                연산자와 두개의 피연산자들을 pop하여 연산하고 연산한 result값을 다시 stack에 넣는것을 반복함.
                이때 while문의 조건에 맞지 않으면 while문은 종료되고 self.stack.pop()을 통해 '('을 stack에서 제거함.
                '''

            elif self.is_operator(char):
            # char가 어떤 연산자인지 여부를 확인하는 조건문
                while not self.is_empty(self.stack) and self.has_higher_precedence(self.stack[-1], char):
                    '''
                    stack이 비어있지 않고, stack의 마지막 요소(연산자)가 char보다 더 높은 우선순위에 있지 않는동안
                    (높은 우선순위에 있을때 동안) 계속 stack에서 연산자를 꺼내어 계산을실행함. 조건을 만족하지 못다면 종료됨.
                    '''
                    operator = self.stack.pop()
                    operand2 = self.stack.pop()
                    operand1 = self.stack.pop()
                    result = self.calculate(operator, operand2, operand1)
                    self.stack.append(result)
                self.stack.append(char)
                # 결과적으로 stack안에 result만 남게 됨.
            
        while not self.is_empty(self.stack):
            operator = self.stack.pop()
            operand2 = self.stack.pop()
            operand1 = self.stack.pop()
            result = self.calculate(operator, operand2, operand1)
            self.stack.append(result)

        return self.stack.pop()
        # while문을 돌다가 연산자와 피연산자가 모두 소진되어 while문의 조건을 충족 시키지 못하면 result가 pop되어 결과값이 반환된다.





import math
'''
math 모듈을 import 함으로써 삼각함수(sin, cos, tan) 계산이 가능하게 하였고,
공학용 계산에서 일반적으로 사용되는 상수를 사용할 수 있어야 한다
'''
import numpy
 #NumPy 라이브러리를 현재 코드에서 사용할 수 있도록 가져오는 명령임. 
import matplotlib.pyplot as plt

class EngineerCalculator(Calculator):
    # 위에서 만든  Calculator 클래스를 상속함.
    def __init__(self):
        super().__init__()
        # 상속된 Calculator클래스의 생성자를 먼저 생성하고 이후에 EngineerCalculator 클래스의 생성자를 생성함.


    def calculate_factorial(self, n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.calculate_factorial(n - 1)
    #팩토리얼 계산을 재귀방식으로 나타냄.
        
    def calculate_trigonometric(self, func, angle):
        if func == 'sin':
            return math.sin(angle)
        elif func == 'cos':
            return math.cos(angle)
        elif func == 'tan':
            return math.tan(angle)
    # 삼각함수를 계산하는 메서드로, 주어진 각도에 대해 각 삼각함수의 값을 반환함.
        
    def calculate_matrix(self, matrix):
        return numpy.linalg.det(matrix)
    '''
    determinant(행렬식)을 계산하는 메서드. 
    NumPy 라이브러리의 numpy.linalg.det 함수를 사용하여 determinant를 계산함.
    '''

    def plot_graph(self, x_values, y_values, title="Graph"):
        plt.plot(x_values, y_values)
        plt.title(title)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()



