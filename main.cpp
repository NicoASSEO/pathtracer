#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0, 1);

double sqr(double x) { return x * x; };

void boxMuller2D(double sigma, double& dx, double& dy) {
	int tid = omp_get_thread_num() % 32;
	double r1 = std::max(uniform(engine[tid]), 1e-12);
	double r2 = uniform(engine[tid]);
	double factor = sigma * std::sqrt(-2.0 * std::log(r1));
	double angle = 2.0 * M_PI * r2;
	dx = factor * std::cos(angle);
	dy = factor * std::sin(angle);
}

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector multiply(const Vector& a, const Vector& b) {
	return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

Vector sampleHemi(const Vector& N) {
	int tid = omp_get_thread_num();
	double r1 = uniform(engine[tid]);
	double r2 = uniform(engine[tid]);

	double phi = 2. * M_PI * r1;
	double x = std::cos(phi) * std::sqrt(r2);
	double y = std::sin(phi) * std::sqrt(r2);
	double z = std::sqrt(1. - r2);

	Vector T;
	if (std::fabs(N[0]) < 0.9) {
		T = cross(Vector(1, 0, 0), N);
	}
	else {
		T = cross(Vector(0, 1, 0), N);
	}
	T.normalize();
	Vector B = cross(N, T);

	Vector wi = x * T + y * B + z * N;
	wi.normalize();
	return wi;
}


class Ray {
public:
	Ray(const Vector& origin, const Vector& unit_direction) : O(origin), u(unit_direction) {};
	Vector O, u;
};

class Object {
public:
	Object(const Vector& albedo, bool mirror = false, bool transparent = false) : albedo(albedo), mirror(mirror), transparent(transparent) {};

	virtual bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const = 0;

	Vector albedo;
	bool mirror, transparent;
};

class Sphere : public Object {
public:
	Sphere(const Vector& center, double radius, const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent), C(center), R(radius) {};

	// returns true iif there is an intersection between the ray and the sphere
	// if there is an intersection, also computes the point of intersection P, 
	// t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
	// and the unit normal N
	bool intersect(const Ray& ray, Vector& P, double &t, Vector& N) const {
		 // TODO (lab 1) : compute the intersection (just true/false at the begining of lab 1, then P, t and N as well)
		 Vector OC = ray.O - this->C;
		 double delta = std::pow(dot(ray.u, OC), 2) - (OC.norm2() - R*R);
		 if (delta < 0) {
			 return false;
			}
		double t1 = dot(ray.u, this->C - ray.O) - std::sqrt(delta);
		double t2 = dot(ray.u, this->C - ray.O) + std::sqrt(delta);
		 if (t1 < 0 && t2 < 0) {
			return false;
		 }
		 if (t1 < 0 && t2 >= 0) {
			t = t2;
		 } else {
			t = t1;
		 }
		 if (t1 >= 0 && t2 >= 0) {
			t = std::min(t1, t2);
		 }
		 P = ray.O + t * ray.u;
		 N = (P - this->C);
		 N.normalize();
		 return true;
	}

	double R;
	Vector C;
};


// I will provide you with an obj mesh loader (labs 3 and 4)
class TriangleMesh : public Object {
public:
	TriangleMesh(const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent) {};

	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const {
		// TODO (labs 3 and 4)
		return false;
	}
};


class Scene {
public:
	Scene() {};
	void addObject(const Object* obj) {
		objects.push_back(obj);
	}

	// returns true iif there is an intersection between the ray and any object in the scene
    // if there is an intersection, also computes the point of the *nearest* intersection P, 
    // t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
    // and the unit normal N. 
	// Also returns the index of the object within the std::vector objects in object_id
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N, int &object_id) const  {

		// TODO (lab 1): iterate through the objects and check the intersections with all of them, 
		double closest_t = std::numeric_limits<double>::infinity();
		bool found = false;
		for (int i = 0; i < objects.size(); i++) {
			Vector closest_P, closest_N;
			double t_temp;
			if (objects[i] -> intersect(ray, closest_P, t_temp, closest_N)) {
				if (t_temp < closest_t) {
					closest_t = t_temp;
					P = closest_P;
					N = closest_N;
					t = t_temp;
					object_id = i;
					found = true;
				}
			}
		}
		return found;
	}


	// return the radiance (color) along ray
	Vector getColor(const Ray& ray, int recursion_depth) {

		if (recursion_depth >= max_light_bounce) return Vector(0, 0, 0);

		// TODO (lab 1) : if intersect with ray, use the returned information to compute the color ; otherwise black
		// in lab 1, the color only includes direct lighting with shadows

		Vector P, N;
		double t;
		int object_id;
		double epsilon = 1e-4;
		if (intersect(ray, P, t, N, object_id)) {

			if (objects[object_id]->mirror) {

				Vector reflected_direction = ray.u - 2 * dot(ray.u, N) * N;
				reflected_direction.normalize();
				Ray reflected_ray(P + epsilon * N, reflected_direction);
				return getColor(reflected_ray, recursion_depth + 1);

				// return getColor in the reflected direction, with recursion_depth+1 (recursively)
			} // else

			if (objects[object_id]->transparent) { // optional

				// return getColor in the refraction direction, with recursion_depth+1 (recursively)
			} // else

			// test if there is a shadow by sending a new ray
			Vector direct(0,0,0);
			Vector to_light = light_position - P;
			double dist_to_light = to_light.norm();
			Vector light_dir = to_light / dist_to_light; 
			Ray shadow_ray(P + epsilon * N, light_dir);

			Vector P_shadow;
			int object_id_shadow;
			Vector N_shadow;
			double t_shadow;
			bool in_shadow = this->intersect(shadow_ray, P_shadow, t_shadow, N_shadow, object_id_shadow) && t_shadow < (this->light_position - P).norm();

			if (!in_shadow) {
				double dist2 = to_light.norm2();
				double Li = light_intensity / (4.0 * M_PI * dist2); 
				double cos_theta = std::max(0.0, dot(N, light_dir));  
				direct = Li * (objects[object_id]->albedo / M_PI) * cos_theta;
		}
			
			// TODO (lab 2) : add indirect lighting component with a recursive call
			Vector wi = sampleHemi(N);
			Ray indirect_ray(P + epsilon * N, wi);
			Vector indirect = multiply(objects[object_id]->albedo, getColor(indirect_ray, recursion_depth + 1));

			return direct + indirect;
		}

		

		return Vector(0, 0, 0);
	}

	std::vector<const Object*> objects;

	Vector camera_center, light_position;
	double fov, gamma, light_intensity;
	int max_light_bounce;
};


int main() {
	int W = 512;
	int H = 512;
	int samples_per_pixel = 32;

	for (int i = 0; i<32; i++) {
		engine[i].seed(i);
	}

	Sphere center_sphere(Vector(0, 0, 0), 10., Vector(0.8, 0.8, 0.8), true);
	Sphere wall_left(Vector(-1000, 0, 0), 940, Vector(0.5, 0.8, 0.1));
	Sphere wall_right(Vector(1000, 0, 0), 940, Vector(0.9, 0.2, 0.3));
	Sphere wall_front(Vector(0, 0, -1000), 940, Vector(0.1, 0.6, 0.7));
	Sphere wall_behind(Vector(0, 0, 1000), 940, Vector(0.8, 0.2, 0.9));
	Sphere ceiling(Vector(0, 1000, 0), 940, Vector(0.3, 0.5, 0.3));
	Sphere floor(Vector(0, -1000, 0), 990, Vector(0.6, 0.5, 0.7));

	Scene scene;
	scene.camera_center = Vector(0, 0, 55);
	scene.light_position = Vector(-10,20,40);
	scene.light_intensity = 2E7;
	scene.fov = 60 * M_PI / 180.;
	scene.gamma = 2.0;    // TODO (lab 1) : play with gamma ; typically, gamma = 2.2
	scene.max_light_bounce = 5;

	scene.addObject(&center_sphere);

	
	scene.addObject(&wall_left);
	scene.addObject(&wall_right);
	scene.addObject(&wall_front);
	scene.addObject(&wall_behind);
	scene.addObject(&ceiling);
	scene.addObject(&floor);

	std::vector<unsigned char> image(W * H * 3, 0);

#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			Vector color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; s++) {
				double dx, dy;
				boxMuller2D(0.45, dx, dy);
				Vector ray_direction(j - W / 2. + 0.5 + dx, H / 2. - i - 0.5 + dy, -W / (2 * tan(scene.fov / 2)));
				ray_direction.normalize();
				Ray ray(scene.camera_center, ray_direction);
				color = color + scene.getColor(ray, 0);
			}
			color = color / samples_per_pixel;

			image[(i * W + j) * 3 + 0] = std::min(255., std::max(0., 255. * std::pow(color[0] / 255., 1. / scene.gamma)));
			image[(i * W + j) * 3 + 1] = std::min(255., std::max(0., 255. * std::pow(color[1] / 255., 1. / scene.gamma)));
			image[(i * W + j) * 3 + 2] = std::min(255., std::max(0., 255. * std::pow(color[2] / 255., 1. / scene.gamma)));
		}
	}
	stbi_write_png("image.png", W, H, 3, &image[0], 0);

	return 0;
}
