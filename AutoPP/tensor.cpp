#include <iostream>

#include <vector>

struct Grad;
struct Dependency;

std::vector<int> operator+(const std::vector<int>& left, const std::vector<int>& right) {
	std::vector<int> result;
	result.reserve(left.size());
	
	for (unsigned int i = 0; i < left.size(); i++) {
		result.push_back(left[i] + right[i]);
	}

	return result;
}

class Tensor {
private:
	std::vector<int> data;
	bool requires_grad;
	std::vector<Dependency> depends_on;
	std::unique_ptr<Tensor> grad;
	int shape;
public:
	Tensor(std::vector<int> data, bool requires_grad = false, std::vector<Dependency> depends_on = {})
		: data(data), requires_grad(requires_grad), depends_on(depends_on) {
		grad = nullptr;
		shape = data.size();

		if (requires_grad)
			zero_grad();
	}

	Tensor(int size) {
		std::vector<int> vec(size, 0);
		data = vec;
		requires_grad = false;
		depends_on = {};
		grad = nullptr;
		shape = shape;
	}

	void zero_grad() {
		grad = std::make_unique<Tensor>(shape);
	}

	void backward(std::unique_ptr<Tensor> grad = nullptr) {
		if (!requires_grad) {
			std::cerr << "Called backward on a non-requires-grad tensor" << std::endl;;
			return;
		}

		if (grad == nullptr) {
			if (shape == 0) {
				grad = std::make_unique<Tensor>(std::vector<int>{ 1 });
			}
			else {
				std::cerr << "Grad must be specified for non-0-tensor";
			}
		}

		this->grad->data = this->grad->data + grad->data;
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
	Tensor(*grad_fn)(Tensor);
};

int main() {
	Tensor t({ 1, 2, 3, 4 }, true);
	
	std::cout << t << std::endl;

	std::unique_ptr<Tensor> grad = std::make_unique<Tensor>(std::vector<int>{ 5, 6, 7, 8 });
	t.backward(std::move(grad));
	//t.backward();

	std::cin.get();
}
