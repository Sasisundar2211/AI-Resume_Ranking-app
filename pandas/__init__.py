"""Lightweight pandas-compatible DataFrame subset."""
class DataFrame:
    def __init__(self, data):
        self.data = data

    def sort_values(self, by, ascending=True):
        keys = self.data.get(by, [])
        idx = sorted(range(len(keys)), key=lambda i: keys[i], reverse=not ascending)
        sorted_data = {k: [vals[i] for i in idx] for k, vals in self.data.items()}
        return DataFrame(sorted_data)

    def __repr__(self):
        return f"DataFrame({self.data})"
