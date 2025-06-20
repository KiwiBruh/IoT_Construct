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
#include "objWork.h"


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


__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, vec3 light_pos, vec3 light_col, trig* trigs, int total_trigs) {
	int k, k_min = -1;
	double ts_min;
	for (k = 0; k < total_trigs; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div;
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if (k_min == -1)
		return { 130, 130, 130, 255 };

	pos = add({ dir.x * ts_min,dir.y * ts_min,dir.z * ts_min }, pos);
	dir = diff(light_pos, pos);
	double length = dot(dir, dir);
	dir = norm(dir);

	for (k = 0; k < total_trigs; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div;
		if (ts > 0.0 && ts < length && k != k_min) {
			return { 0, 0, 0, 0 };
		}
	}


	uchar4 k_min_col = { (uchar)(trigs[k_min].color.x * light_col.x), (uchar)(trigs[k_min].color.y * light_col.y), (uchar)(trigs[k_min].color.z * light_col.z), 0 };
	return k_min_col;
}

__host__ __device__ bool check_visibility(vec3 sensor_pos, vec3 sensor_dir, double fov_angle, trig* trigs,
	int total_trigs, const char* target_obj) {
	int total_doors = 0;
	int visible_doors = 0;
	for (int i = 0; i < total_trigs; i++) {
		if (strcmp(trigs[i].object, target_obj) == 0) {
			total_doors++;
		}
	}

	if (total_doors == 0) return true; // Если дверей нет, считаем размещение хорошим

	int rays_per_dim = 10; 
	double angle_step = fov_angle / (rays_per_dim - 1);

	vec3 forward = norm(sensor_dir);
	vec3 up = { 0.0, 0.0, 1.0 };
	vec3 right = norm(prod(forward, up));
	up = norm(prod(right, forward));

	for (int i = 0; i < total_trigs; i++) {
		if (strcmp(trigs[i].object, target_obj) != 0) continue;

		vec3 tri_center = {
			(trigs[i].a.x + trigs[i].b.x + trigs[i].c.x) / 3.0,
			(trigs[i].a.y + trigs[i].b.y + trigs[i].c.y) / 3.0,
			(trigs[i].a.z + trigs[i].b.z + trigs[i].c.z) / 3.0
		};

		vec3 to_tri = diff(tri_center, sensor_pos);
		double dist_to_tri = length(to_tri);
		vec3 dir_to_tri = norm(to_tri);

		double angle = acos(dot(forward, dir_to_tri)) * 180.0 / M_PI;
		if (angle > fov_angle / 2.0) continue;

		bool is_visible = true;
		for (int k = 0; k < total_trigs; k++) {
			if (k == i) continue;

			vec3 e1 = diff(trigs[k].b, trigs[k].a);
			vec3 e2 = diff(trigs[k].c, trigs[k].a);
			vec3 p = prod(dir_to_tri, e2);
			double div = dot(p, e1);
			if (fabs(div) < 1e-10) continue;

			vec3 t = diff(sensor_pos, trigs[k].a);
			double u = dot(p, t) / div;
			if (u < 0.0 || u > 1.0) continue;

			vec3 q = prod(t, e1);
			double v = dot(q, dir_to_tri) / div;
			if (v < 0.0 || v + u > 1.0) continue;

			double ts = dot(q, e2) / div;
			if (ts > 0.0 && ts < dist_to_tri) {
				is_visible = false;
				break;
			}
		}

		if (is_visible) {
			visible_doors++;
		}
	}

	return (visible_doors * 5 >= total_doors);
}



__host__ __device__ void ssaa(uchar4* data, int w, int h, int ssaa_multiplier, uchar4* ssaa_data, int n, int m) {
	int4 soften = { 0, 0, 0, 0 };
	for (int j = 0; j < ssaa_multiplier; j++) {
		for (int i = 0; i < ssaa_multiplier; i++) {
			int index = n * w * ssaa_multiplier * ssaa_multiplier + m * ssaa_multiplier + j * w * ssaa_multiplier + i;
			soften.x += ssaa_data[index].x;
			soften.y += ssaa_data[index].y;
			soften.z += ssaa_data[index].z;
		}
	}
	data[n * w + m].x = (uchar)(int)(soften.x / (ssaa_multiplier * ssaa_multiplier));
	data[n * w + m].y = (uchar)(int)(soften.y / (ssaa_multiplier * ssaa_multiplier));
	data[n * w + m].z = (uchar)(int)(soften.z / (ssaa_multiplier * ssaa_multiplier));
}

__global__ void ssaa_device(uchar4* data, int w, int h, int ssaa_multiplier, uchar4* ssaa_data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int n = idy; n < h; n += offsety) {
		for (int m = idx; m < w; m += offsetx) {
			ssaa(data, w, h, ssaa_multiplier, ssaa_data, n, m);
		}
	}
}

__global__ void render_device(vec3 pc, vec3 pv, int w, int h, double angle, uchar4* data, vec3 light_pos, vec3 light_col, trig* trigs_arr, int total_trigs) {
	int i, j;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
	vec3 by = norm(prod(bx, bz));
	for (i = idx; i < w; i += offsetx)
		for (j = idy; j < h; j += offsety) {
			vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
			vec3 dir = mult(bx, by, bz, v);
			data[(h - 1 - j) * w + i] = ray(pc, norm(dir), light_pos, light_col, trigs_arr, total_trigs);
			/*			print(pc);
						print(add(pc, dir));
						printf("\n\n\n"); */
		}
	/*	print(pc);
		print(pv);
		printf("\n\n\n");*/
}



void render_host(vec3 pc, vec3 pv, int w, int h, double angle, uchar4* data, vec3 light_pos, vec3 light_col, trig* trigs_arr, int total_trigs) {
	int i, j;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
	vec3 by = norm(prod(bx, bz));
	for (i = 0; i < w; i++)
		for (j = 0; j < h; j++) {
			vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
			vec3 dir = mult(bx, by, bz, v);
			data[(h - 1 - j) * w + i] = ray(pc, norm(dir), light_pos, light_col, trigs_arr, total_trigs);
			/*			print(pc);
						print(add(pc, dir));
						printf("\n\n\n"); */
		}
	/*	print(pc);
		print(pv);
		printf("\n\n\n");*/
}
int main(int argc, char* argv[]) {
	string key;
	cin >> key;
	if (argv[1])
		key = argv[1];
	bool gpu = true;

	if (argc == 1 || key == "--gpu")
		gpu = true;

	if (key == "--cpu")
		gpu = false;

	if (key == "--default") {
		cout << "120\n"
			"./res\n"
			"640 480 120\n"
			"7.0 3.0 0.0    2.0 1.0    2.0 6.0 1.0     0.0 0.0\n"
			"2.0 0.0 0.0    0.5 0.1    1.0 4.0 1.0     0.0 0.0\n"
			"4.0 4.0 0.0    1.0 0.0 1.0    2.0     0.0 0.0 0.0\n"
			"1.0 1.0 0.0    1.0 1.0 0.0    2.0     0.0 0.0 0.0\n"
			"-2.5 -2.5 0.0    0.0 1.0 1.0    2.0     0.0 0.0 0.0\n"
			"-10.0 -10.0 -1.0    -10.0 10.0 -1.0    10.0 10.0 -1.0    10.0 -10.0 -1.0    ./folder    1.0 0.5 0.5    0.5\n"
			"1\n"
			"100 100 100    1.0 1.0 1.0\n"
			"1 3\n";
		return 0;
	}

	string other;


	int fr = 10;
	string path = "./res";
	string MtlPath = "./Aparts/new3.mtl";
	string ObjPath = "./Aparts/new3.obj";
	int k, w = 1024, h = 768, angle = 120;

	double r_0c = 350.0, z_0c = 250.0, phi_0c = 0.0;
	double A_rc = 2.0, A_zc = 1.0;
	double w_rc = 2.0, w_zc = 6.0, w_phic = 1.0;
	double p_rc = 0.0, p_zc = 0.0;

	double r_0v = 2.0, z_0v = 0.0, phi_0v = 0.0;
	double A_rv = 0.5, A_zv = 0.1;
	double w_rv = 1.0, w_zv = 4.0, w_phiv = 1.0;
	double p_rv = 0.0, p_zv = 0.0;

	int n_lights = 1;
	int recursion_step = 1;
	int ssaa_multiplier = 3;

	vec3 light_pos = { 0, 0, 500 }, light_col = { 1.0, 1.0, 1.0 };
	
	int ssaa_w = w * ssaa_multiplier;
	int ssaa_h = h * ssaa_multiplier;
	int rays = ssaa_w * ssaa_h;

	auto materials = parseMTL(MtlPath);
	vector<trig> trigs = parseOBJ(ObjPath, materials);
	trig* trigs_arr;
	uchar4* data = nullptr;
	uchar4* data_ssaa = nullptr;
	data = new uchar4[rays];
	data_ssaa = new uchar4[rays];

	device MotionSensor;//в будущем будет отдельная библиотека устройств
	MotionSensor.type = 1;
	MotionSensor.fov = 180;
	MotionSensor.ObjPath = "D:/diplom/smoke_detector.obj";
	MotionSensor.MtlPath = "D:/diplom/smoke_detector.mtl";
	MotionSensor.placement = (124,-29,118);
	MotionSensor.dir = (124,-30,118);

	if(MotionSensor.type == 1){
		bool good_placement = check_visibility(MotionSensor.placement, MotionSensor.dir, MotionSensor.fov, trigs,total_trigs, "door");

		if (!good_placement) {
			printf("bad placement\n");
		}
		else {
			addToScene(MotionSensor, "./Aparts/new3.obj");
		}
	}



	trigs_arr = trigs.data();
	int total = trigs.size();
	vec3 pc, pv;
	double r_c, z_c, phi_c, r_v, z_v, phi_v, time = 0.0;
	for (k = 0; k < fr; k++) {
		double cur_time = k * 2.0 * M_PI / fr;
		r_c = r_0c + A_rc * sin(w_rc * cur_time + p_rc);
		z_c = z_0c + A_zc * sin(w_zc * cur_time + p_zc);
		phi_c = phi_0c + w_phic * cur_time;
		r_v = r_0v + A_rv * sin(w_rv * cur_time + p_rv);
		z_v = z_0v + A_zv * sin(w_zv * cur_time + p_zv);
		phi_v = phi_0v + w_phiv * cur_time;
		pc = { r_c * cos(phi_c), r_c * sin(phi_c), z_c };
		pv = { r_v * cos(phi_v), r_v * sin(phi_v), z_v };

		auto start = chrono::steady_clock::now();


		if (gpu == true) {
			uchar4* gpu_data;
			uchar4* gpu_data_ssaa;
			trig* gpu_trigs;


			CSC(cudaMalloc(&gpu_data, w * h * sizeof(uchar4)));
			CSC(cudaMemcpy(gpu_data, data, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

			CSC(cudaMalloc(&gpu_data_ssaa, ssaa_w * ssaa_h * sizeof(uchar4)));
			CSC(cudaMemcpy(gpu_data_ssaa, data_ssaa, ssaa_w * ssaa_h * sizeof(uchar4), cudaMemcpyHostToDevice));


			CSC(cudaMalloc(&gpu_trigs, total * sizeof(trig)));
			cudaMemcpy(gpu_trigs, trigs_arr, total * sizeof(trig), cudaMemcpyHostToDevice);


			render_device << < 1024, 768 >> > (pc, pv, ssaa_w, ssaa_h, angle, gpu_data_ssaa, light_pos, light_col, gpu_trigs, total);
			CSC(cudaGetLastError());

			ssaa_device << < 1024, 768 >> > (gpu_data, w, h, ssaa_multiplier, gpu_data_ssaa);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(data, gpu_data, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));


			CSC(cudaFree(gpu_data));
			CSC(cudaFree(gpu_data_ssaa));
			CSC(cudaFree(gpu_trigs));
		}
		else {
			cout << "not a gpu";
			//render_host(pc, pv, ssaa_w, ssaa_h, angle, data_ssaa, light_pos, light_col, trigs_arr, trigs.size());
			//ssaa_host(data, w, h, ssaa_multiplier, data_ssaa);
		}

		auto end = chrono::steady_clock::now();
		cout << "Iteration " << k + 1 << " of " << fr << "\t|\t";
		double iteration_time = ((double)chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0;
		time += iteration_time;
		cout << "time " << iteration_time << "ms\t|\t";
		cout << "rays " << rays << "\t|\n";

		string buff = path + "/" + to_string(k) + ".data";
		//cout << buff << endl;
		FILE* out = fopen(buff.c_str(), "wb");
		fwrite(&w, sizeof(int), 1, out);
		fwrite(&h, sizeof(int), 1, out);
		fwrite(data, sizeof(uchar4), w * h, out);
		fclose(out);
	}
	cout << "Total trigs: " << total << ". Frame size: " << w << "x" << h << "\n";
	cout << "Total time: " << time << "\t" << "Total frames: " << fr << "\n";
	delete[] data;
	return 0;
}
