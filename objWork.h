#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "polygons.h"
#include <set>
#include <queue>
#include <Eigen/Dense>


using namespace std;

#define M_PI 3.14159265358979323846

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)


// Плоскость (для QEM)
struct Plane {
	double a, b, c, d; // Уравнение плоскости: ax + by + cz + d = 0
};

struct Quadric {
	Matrix4f A;

	Quadric() : A(Matrix4f::Zero()) {}

	// Добавляем плоскость в квадрику
	void addPlane(const Plane& p) {
		Vector4f v(p.a, p.b, p.c, p.d);
		A += v * v.transpose();
	}

	// Объединяем квадрики
	void merge(const Quadric& other) {
		A += other.A;
	}
};

struct Vertex {
	vec3 pos;
	Quadric q; // QEM для вершины
	bool


// Ребро для схлопывания
struct Edge {
	int v1, v2;       // Индексы вершин
	double cost;       // Ошибка схлопывания
	vec3 new_pos;     // Позиция новой вершины

	bool operator<(const Edge& other) const {
		return cost > other.cost; // Для min-heap
	}
};


float QError(const Vertex& v, const vec3& new_pos) {
	float error = 0.0f;
	Plane plane;
	for (int i = 0; i < v.planes.size(); i++) {
		plane = v.planes[i];
		float dist = plane.a * new_pos.x + plane.b * new_pos.y + plane.c * new_pos.z + plane.d;
		error += dist * dist;
	}
	return error;
}

// Вычисляет уравнение плоскости по трём точкам
Plane computePlane(const vec3& a, const vec3& b, const vec3& c) {
	vec3 ab = { b.x - a.x, b.y - a.y, b.z - a.z };
	vec3 ac = { c.x - a.x, c.y - a.y, c.z - a.z };

	vec3 normal = {
		ab.y * ac.z - ab.z * ac.y,
		ab.z * ac.x - ab.x * ac.z,
		ab.x * ac.y - ab.y * ac.x
	};

	double len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	normal.x /= len;
	normal.y /= len;
	normal.z /= len;

	return { normal.x, normal.y, normal.z, -(normal.x * a.x + normal.y * a.y + normal.z * a.z) };
}

// Вычисляет оптимальную позицию для новой вершины при схлопывании
vec3 computeOptimalPosQEM(const Vertex& v1, const Vertex& v2) {
	// Объединённая квадрика
	Quadric Q = v1.q;
	Q.merge(v2.q);

	// Создаём систему уравнений для нахождения оптимальной позиции
	Matrix4f A = Q.A;

	// Фиксируем последнюю строку для решения [v_new, 1]
	A.row(3) << 0, 0, 0, 1;

	Vector4f b(0, 0, 0, 1); // Правая часть: [0, 0, 0, 1]

	// Решаем систему A·x = b
	Vector4f x = A.colPivHouseholderQr().solve(b);

	// Если решение найдено, возвращаем новую позицию
	if ((A * x - b).norm() < 1e-6) {
		return { x(0), x(1), x(2) };
	}

	// Если система вырождена, возвращаем середину ребра
	return {
		(v1.pos.x + v2.pos.x) * 0.5f,
		(v1.pos.y + v2.pos.y) * 0.5f,
		(v1.pos.z + v2.pos.z) * 0.5f
	};
}

// Добавляет ребро в очередь
void addEdgeToQueue(priority_queue<Edge>& queue, const vector<Vertex>& vertices,
	const map<vec3, int>& pos_to_idx, const vec3& a, const vec3& b) {
	int v1 = pos_to_idx.at(a);
	int v2 = pos_to_idx.at(b);

	Edge edge;
	edge.v1 = v1;
	edge.v2 = v2;

	// Оптимальная позиция новой вершины (минимизируем QEM)
	edge.new_pos = computeOptimalPos(vertices[v1], vertices[v2]);

	// Стоимость схлопывания
	edge.cost = QError(vertices[v1], edge.new_pos) + QError(vertices[v2], edge.new_pos);

	queue.push(edge);
}

// Схлопывает ребро
void collapseEdge(std::vector<Vertex>& vertices, const Edge& edge) {
	vertices[edge.v1].pos = edge.new_pos;
	vertices[edge.v1].planes.insert(vertices[edge.v1].planes.end(),
		vertices[edge.v2].planes.begin(),
		vertices[edge.v2].planes.end());
	vertices[edge.v2].valid = false; // Помечаем как удалённую
}

std::vector<trig> simplifyMesh(const std::vector<trig>& mesh, int target_tri_count) {
	// 1. Собираем вершины и плоскости
	std::vector<Vertex> vertices;
	std::map<vec3, int> pos_to_idx;

	for (const auto& tri : mesh) {
		Plane p = computePlane(tri.a, tri.b, tri.c);

		for (const auto& v_pos : { tri.a, tri.b, tri.c }) {
			if (!pos_to_idx.count(v_pos)) {
				pos_to_idx[v_pos] = vertices.size();
				vertices.push_back({ v_pos, {p} });
			}
			else {
				vertices[pos_to_idx[v_pos]].planes.push_back(p);
			}
		}
	}

	// 2. Строим приоритетную очередь рёбер
	std::priority_queue<Edge> edge_queue;

	for (const auto& tri : mesh) {
		addEdgeToQueue(edge_queue, vertices, pos_to_idx, tri.a, tri.b);
		addEdgeToQueue(edge_queue, vertices, pos_to_idx, tri.b, tri.c);
		addEdgeToQueue(edge_queue, vertices, pos_to_idx, tri.c, tri.a);
	}

	// 3. Схлопываем рёбра, пока не достигнем нужного количества полигонов
	while (mesh.size() > target_tri_count && !edge_queue.empty()) {
		Edge edge = edge_queue.top();
		edge_queue.pop();

		// Схлопываем ребро (v1, v2) -> v_new
		collapseEdge(vertices, edge);

	}

	// 4. Перестраиваем меш
	return rebuildMesh(vertices);
}

map<string, Material> parseMTL(const string& filePath) {
	map<string, Material> materials;
	ifstream file(filePath);
	string line;
	Material currentMaterial;
	vec3 Kd;
	float d = 1.0;

	while (getline(file, line)) {
		istringstream iss(line);
		string prefix;
		iss >> prefix;

		if (prefix == "newmtl") {
			// Save previous material (if any) and start a new one
			if (!currentMaterial.name.empty()) {
				materials[currentMaterial.name] = currentMaterial;
			}
			currentMaterial = Material();  // Reset
			iss >> currentMaterial.name;
		}
		else if (prefix == "d") {
			iss >> d;
		}
		else if (prefix == "Kd") {
			iss >> Kd.x >> Kd.y >> Kd.z;
			currentMaterial.color = incolor(Kd, d);
		}
		/*else if (prefix == "Ka") {
			iss >> currentMaterial.Kd[0] >> currentMaterial.Kd[1] >> currentMaterial.Kd[2];
		}
		else if (prefix == "Ks") {
			iss >> currentMaterial.Ks[0] >> currentMaterial.Ks[1] >> currentMaterial.Ks[2];
		}
		else if (prefix == "Ns") {
			iss >> currentMaterial.Ns;
		}*/
	}

	// Add the last material
	if (!currentMaterial.name.empty()) {
		materials[currentMaterial.name] = currentMaterial;
	}
	file.close();
	return materials;
}

vector<trig> parseOBJ(const string& filePath, map<string, Material>& materials) {
	vector<trig> trigs;
	ifstream file(filePath);
	string line;
	vector<vec3> vertices;
	vector<string> faceMaterials;
	string currentMaterial = "";
	vec3 currentVertice;
	int firstIndex;


	while (getline(file, line)) {
		istringstream iss(line);
		string prefix;
		iss >> prefix;

		if (prefix == "v") {
			iss >> currentVertice.y >> currentVertice.z >> currentVertice.x;
			vertices.push_back(currentVertice);
		}
		else if (prefix == "usemtl") {
			iss >> currentMaterial;
		}
		else if (prefix == "f") {
			vector<int> face;
			string vertex;
			while (iss >> vertex) {
				size_t pos = vertex.find('/');
				if (pos != string::npos) {
					vertex = vertex.substr(0, pos);
				}
				if (trigs.empty() && face.empty())
					firstIndex = std::stoi(vertex);
				face.push_back(std::stoi(vertex) - firstIndex);
				//face.push_back(std::stoi(vertex) - 1); // .obj использует 1-based индексы
			}
			trigs.push_back({ vertices[face[0]] , vertices[face[1]], vertices[face[2]], materials[currentMaterial].color });
		}
	}
	file.close();
	simplifyMesh(trigs, ((int)(trigs.size()) / 2));
	return trigs;
}

void appendObjFile(const string& destPath, const string& srcPath) {
	ofstream destFile(destPath, ios::app);
	destFile << "\n";
	string line;
	while (getline(srcFile, line)) {
		if (line.empty() && destFile.tellp() == 0) {
			continue;
		}
		destFile << line << "\n";
	}

	srcFile.close();
	destFile.close();

	cout << "Successfully appended " << srcPath << " to " << destPath << endl;
}

void addToScene(device sensor, const string& destPath) {
	appendObjFile(const string & destPath, device.ObjPath);
}

