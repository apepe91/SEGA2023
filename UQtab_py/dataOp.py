class Dati:
    @staticmethod
    def min(numbers):
        if len(numbers) == 0:
            return 0
        min_val = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] < min_val:
                min_val = numbers[i]
        return min_val

    @staticmethod
    def max(numbers):
        if len(numbers) == 0:
            return 0
        max_val = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] > max_val:
                max_val = numbers[i]
        return max_val

    @staticmethod
    def sum(x):
        if len(x) == 0:
            return 0
        total_sum = x[0]
        correction = 0
        transition = 0
        for i in range(1, len(x)):
            transition = total_sum + x[i]
            if abs(total_sum) >= abs(x[i]):
                correction += total_sum - transition + x[i]
            else:
                correction += x[i] - transition + total_sum
            total_sum = transition
        return total_sum + correction

    @staticmethod
    def sortElements(data):
        sorted_data = data[:]
        for i in range(len(sorted_data)):
            tmp = sorted_data[i]
            j = i - 1
            while j >= 0 and sorted_data[j] > tmp:
                sorted_data[j + 1] = sorted_data[j]
                j -= 1
            sorted_data[j + 1] = tmp
        return sorted_data
