
# mean

def mean(data):
    if len(data) == 0:
        return None
    return sum(data) / len(data)


# median

def median(data):
    n = len(data)
    if n == 0:
        return None
    sorted_data = sorted(data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid -1] + sorted_data[mid]) /2
    else :
        return sorted_data[mid]


# mode

def mode(data):
    if len(data) == 0:
        return None
    counts = {}
    for num in data:
        counts[num] = counts.get(num, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    if len(modes) == len(counts):
        return None
    return modes


# variance

def variance(data):
    n = len(data)
    if n == 0:
        return None
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / n


# standard deviation

def std(data):
    var = variance(data)
    if var is None:
        return None
    return var ** 0.5



# min and max value

def min_value(data):
    return min(data) if data else None

def max_value(data):
    return max(data) if data else None


# range

def range_value(data):
    if not data:
        return None
    return max(data) - min(data)


# quantiles

def percentile(data, q):
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (q / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + (sorted_data[f+1] - sorted_data[f]) * c
    else:
        return sorted_data[f]





data = [12, 15, 12, 18, 20, 15, 17, 14, 19, 12]



print("Mean:", mean(data))
print("Median:", median(data))
print("Mode:", mode(data))
print("Variance:", variance(data))
print("Standard Deviation:", std(data))
print("Min:", min_value(data))
print("Max:", max_value(data))
print("Range:", range_value(data))
print("25th Percentile:", percentile(data, 25))
print("50th Percentile:", percentile(data, 50))
print("75th Percentile:", percentile(data, 75))
