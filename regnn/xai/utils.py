def smart_number_format(x):
    return f"{x:.2g}" if abs(x) > 10 else f"{x:.2f}"
