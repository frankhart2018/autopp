#include <iostream>

#include <vector>

struct Grad;
struct Dependency;

class Tensor {
private:
	std::vector<int> data;
	bool requires_grad;
	std::vector<Dependency> depends_on;
	std::unique_ptr<Grad> grad;
	int shape;
public:
	Tensor(std::vector<int> data, bool requires_grad = false, std::vector<Dependency> depends_on = {})
		: data(data), requires_grad(requires_grad), depends_on(depends_on) {
		grad = nullptr;
		shape = data.size();

		if (requires_grad)
			zero_grad();
	}

	void zero_grad() {
		grad = std::make_unique<Grad>(data);
	}

	friend std::ostream& operator<<(std::ostream& stream, const Tensor& t) {
		stream << "Tensor([";
		for (int i = 0; i < t.shape - 1; i++)
			stream << t.data[i] << ", ";
		stream << t.data[t.shape - 1] << "], requires_grad=" << std::boolalpha << t.requires_grad << ")";
		return stream;
	}
};

struct Dependency {
private:
	Tensor* tensor;
};

struct Grad {
private:
	std::unique_ptr<Tensor> tensor;
public:
	Grad() {}

	Grad(std::vector<int>& data) {
		std::vector<int> tensor_data(data.size(), 0);
		tensor = std::make_unique<Tensor>(tensor_data);
	}
};

int main() {
	Tensor t({ 1, 2, 3, 4 }, true);
	
	std::cout << t << std::endl;

	std::cin.get();
}
