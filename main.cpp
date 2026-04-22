#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <map>
#include <string>
#include <fstream>

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
	int tid = omp_get_thread_num() % 32;
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


// Class only used in labs 3 and 4 
class TriangleIndices {
	public:
		TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1) {
			vtx[0] = vtxi; vtx[1] = vtxj; vtx[2] = vtxk;
			uv[0] = uvi; uv[1] = uvj; uv[2] = uvk;
			n[0] = ni; n[1] = nj; n[2] = nk;
			this->group = group;
		};
		int vtx[3]; // indices within the vertex coordinates array
		int uv[3];  // indices within the uv coordinates array
		int n[3];   // indices within the normals array
		int group;  // face group
	};

// Class only used in labs 3 and 4 
class TriangleMesh : public Object {
public:
	TriangleMesh(const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent) {};

	// first scale and then translate the current object
	void scale_translate(double s, const Vector& t) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] = vertices[i] * s + t;
		}
	}

	// read an .obj file
	void readOBJ(const char* obj) {
		std::ifstream f(obj);
		if (!f) return;

		std::map<std::string, int> mtls;
		int curGroup = -1, maxGroup = -1;

		// OBJ indices are 1-based and can be negative (relative), this normalizes them
		auto resolveIdx = [](int i, int size) {
			return i < 0 ? size + i : i - 1;
		};

		auto setFaceVerts = [&](TriangleIndices& t, int i0, int i1, int i2) {
			t.vtx[0] = resolveIdx(i0, vertices.size());
			t.vtx[1] = resolveIdx(i1, vertices.size());
			t.vtx[2] = resolveIdx(i2, vertices.size());
		};
		auto setFaceUVs = [&](TriangleIndices& t, int j0, int j1, int j2) {
			t.uv[0] = resolveIdx(j0, uvs.size());
			t.uv[1] = resolveIdx(j1, uvs.size());
			t.uv[2] = resolveIdx(j2, uvs.size());
		};
		auto setFaceNormals = [&](TriangleIndices& t, int k0, int k1, int k2) {
			t.n[0] = resolveIdx(k0, normals.size());
			t.n[1] = resolveIdx(k1, normals.size());
			t.n[2] = resolveIdx(k2, normals.size());
		};

		std::string line;
		while (std::getline(f, line)) {
			// Trim trailing whitespace
			line.erase(line.find_last_not_of(" \r\t\n") + 1);
			if (line.empty()) continue;

			const char* s = line.c_str();

			if (line.rfind("usemtl ", 0) == 0) {
				std::string matname = line.substr(7);
				auto result = mtls.emplace(matname, maxGroup + 1);
				if (result.second) {
					curGroup = ++maxGroup;
				} else {
					curGroup = result.first->second;
				}
			} else if (line.rfind("vn ", 0) == 0) {
				Vector v;
				sscanf(s, "vn %lf %lf %lf", &v[0], &v[1], &v[2]);
				normals.push_back(v);
			} else if (line.rfind("vt ", 0) == 0) {
				Vector v;
				sscanf(s, "vt %lf %lf", &v[0], &v[1]);
				uvs.push_back(v);
			} else if (line.rfind("v ", 0) == 0) {
				Vector pos, col;
				if (sscanf(s, "v %lf %lf %lf %lf %lf %lf", &pos[0], &pos[1], &pos[2], &col[0], &col[1], &col[2]) == 6) {
					for (int i = 0; i < 3; i++) col[i] = std::min(1.0, std::max(0.0, col[i]));
					vertexcolors.push_back(col);
				} else {
					sscanf(s, "v %lf %lf %lf", &pos[0], &pos[1], &pos[2]);
				}
				vertices.push_back(pos);
			}
			else if (line[0] == 'f') {
				int i[4], j[4], k[4], offset, nn;
				const char* cur = s + 1;
				TriangleIndices t;
				t.group = curGroup;

				// Try each face format: v/vt/vn, v/vt, v//vn, v
				if ((nn = sscanf(cur, "%d/%d/%d %d/%d/%d %d/%d/%d%n", &i[0], &j[0], &k[0], &i[1], &j[1], &k[1], &i[2], &j[2], &k[2], &offset)) == 9) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d/%d %d/%d %d/%d%n", &i[0], &j[0], &i[1], &j[1], &i[2], &j[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]);
				} else if ((nn = sscanf(cur, "%d//%d %d//%d %d//%d%n", &i[0], &k[0], &i[1], &k[1], &i[2], &k[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d %d %d%n", &i[0], &i[1], &i[2], &offset)) == 3) {
					setFaceVerts(t, i[0], i[1], i[2]);
				}
				else continue;

				indices.push_back(t);
				cur += offset;

				// Fan triangulation for polygon faces (4+ vertices)
				while (*cur && *cur != '\n') {
					TriangleIndices t2;
					t2.group = curGroup;
					if ((nn = sscanf(cur, " %d/%d/%d%n", &i[3], &j[3], &k[3], &offset)) == 3) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d/%d%n", &i[3], &j[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]);
					} else if ((nn = sscanf(cur, " %d//%d%n", &i[3], &k[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d%n", &i[3], &offset)) == 1) {
						setFaceVerts(t2, i[0], i[2], i[3]);
					} else { 
						cur++; 
						continue; 
					}

					indices.push_back(t2);
					cur += offset;
					i[2] = i[3]; j[2] = j[3]; k[2] = k[3];
				}
			}
		}
	}
	
	
	// TODO ray-mesh intersection (labs 3 and 4)
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const {
		// TODO (labs 3 and 4)
		// lab 3 : once done, speed it up by first checking against the mesh bounding box
		Vector bbox_min = vertices[0], bbox_max = vertices[0];
  	for (const auto& v : vertices) {
      for (int k = 0; k < 3; k++) {
          bbox_min[k] = std::min(bbox_min[k], v[k]);
          bbox_max[k] = std::max(bbox_max[k], v[k]);
      }
  	}

		double t_enter = -std::numeric_limits<double>::infinity();
		double t_exit  =  std::numeric_limits<double>::infinity();
		for (int axis = 0; axis < 3; axis++) {
			double t0 = (bbox_min[axis] - ray.O[axis]) / ray.u[axis];
			double t1 = (bbox_max[axis] - ray.O[axis]) / ray.u[axis];
			if (t0 > t1) std::swap(t0, t1);
			t_enter = std::max(t_enter, t0);
			t_exit  = std::min(t_exit,  t1);
		}
		if (t_enter > t_exit || t_exit < 0) return false;
		
		// lab 3 : for each triangle, compute the ray-triangle intersection with Moller-Trumbore algorithm
		bool hit = false;
		double t_best = std::numeric_limits<double>::infinity();
		for (auto& tri : indices) {
			const Vector& A = vertices[tri.vtx[0]];
			const Vector& B = vertices[tri.vtx[1]];
			const Vector& C = vertices[tri.vtx[2]];
			Vector e1 = B - A;
			Vector e2 = C - A;
			Vector new_normal = cross(e1, e2);
			double denom = dot(ray.u, new_normal);
			Vector AO    = A - ray.O;
			Vector AOxu  = cross(AO, ray.u);
			double beta  =  dot(e2, AOxu) / denom;
			double gamma = -dot(e1, AOxu) / denom;
			double alpha = 1.0 - beta - gamma;
			double t_hit = dot(AO, new_normal) / denom;
			if (t_hit < 0 || t_hit >= t_best || alpha < 0 || alpha > 1 || beta < 0 || beta > 1 || gamma < 0 || gamma > 1) continue;
			t_best = t_hit;
			hit = true;
			P = ray.O + t_best * ray.u;
			N = new_normal;
			N.normalize();		
		}
		t = t_best;
		return hit;
		// lab 4 : recursively apply the bounding-box test from a BVH datastructure
	};



std::vector<TriangleIndices> indices;
std::vector<Vector> vertices;
std::vector<Vector> normals;
std::vector<Vector> uvs;
std::vector<Vector> vertexcolors;
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
	int samples_per_pixel = 16;

	for (int i = 0; i<32; i++) {
		engine[i].seed(i);
	}

	TriangleMesh cat(Vector(0.8, 0.8, 0.8), false, false);
	cat.readOBJ("cat.obj");
	cat.scale_translate(0.6, Vector(0, -10, 0));
	// Sphere center_sphere(Vector(0, 0, 0), 10., Vector(0.8, 0.8, 0.8), true);
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
	
	// scene.addObject(&center_sphere);
	
	
	scene.addObject(&cat);
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
