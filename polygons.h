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


using namespace std;

typedef unsigned char uchar;

struct vec3 {
	double x;
	double y;
	double z;
};

struct Material {
	string name;
	//vec3 Ka;		// Ambient color (RGB)
	//float Kd[3];  // Diffuse color (RGB)
	//float Ks[3];  // Specular color (RGB)
	//float Ns;     // Shininess
	//float d;		// Transparency (dissolve)
	uchar4 color;
};



__host__ __device__ double dot(vec3 a, vec3 b) {//скалярное произведение
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*
| i  j  k  |
| x1 y1 z1 | = i * (y1 * z2 - z1 * y2) + j * ... + k * ...
| x2 y2 z2 |
*/

__host__ __device__ vec3 prod(vec3 a, vec3 b) {//векторное произведение
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,  a.x * b.y - a.y * b.x };
}

__host__ __device__ vec3 norm(vec3 v) {//нормировка
	double l = sqrt(dot(v, v));
	return { v.x / l, v.y / l, v.z / l };
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {//вычитание
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {//сложение
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {//умножение a b c на вектор v
	return { a.x * v.x + b.x * v.y + c.x * v.z,
			a.y * v.x + b.y * v.y + c.y * v.z,
			a.z * v.x + b.z * v.y + c.z * v.z };
}

void print(vec3 v) {
	printf("%e %e %e\n", v.x, v.y, v.z);
}



struct trig {//один полигон
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
	string object;
};

__host__ __device__ uchar4 incolor(vec3 color, float d = 1.0) {
	uchar4 colorc = { (uchar)(int)(color.x * 255.),(uchar)(int)(color.y * 255.),(uchar)(int)(color.z * 255.),(uchar)(int)(d * 255.) };
	return colorc;
};


struct device {
	int type; //1 - датчик движения
	int fov;
	string ObjPath;
	string MtlPath;
	vec3 placement;
	vec3 dir;
};
