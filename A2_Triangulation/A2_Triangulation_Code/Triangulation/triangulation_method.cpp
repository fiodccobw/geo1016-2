/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>


using namespace easy3d;

/// convert a 3 by 3 matrix of type 'Matrix<double>' to mat3
mat3 to_mat3(Matrix<double> &M) {
    mat3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}


/// convert M of type 'matN' (N can be any positive integer) to type 'Matrix<double>'
template<typename mat>
Matrix<double> to_Matrix(const mat &M) {
    const int num_rows = M.num_rows();
    const int num_cols = M.num_columns();
    Matrix<double> result(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

void getM(Matrix<double>& K,
          Matrix<double>& R,
          std::vector<double>& t,
          Matrix<double>& M1,
          Matrix<double>& M2) {
    Matrix<double> m1(3, 4, 0.0);
    Matrix<double> m2(3, 4, 0.0);
    Matrix<double> E1(3, 4, 0.0);
    Matrix<double> E2(3, 4, 0.0);
    for (int i = 0; i < 3; i++) {
        E1[i][i] = 1;
    }
    E2.set_column(R.get_column(0), 0);
    E2.set_column(R.get_column(1), 1);
    E2.set_column(R.get_column(2), 2);
    E2.set_column(t, 3);

    //std::cout << "extrinsic" << R << std::endl;
    //std::cout << "extrinsic" << t << std::endl;
    //std::cout << "extrinsic" << E2 << std::endl;
    m1 = K * E1;
    m2 = K * E2;
    M1 = m1;
    M2 = m2;
}

float dis_square(vec3 &pt0, vec3 &pt1){
    return (pt0.x - pt1.x) * (pt0.x - pt1.x) + (pt0.y - pt1.y) * (pt0.y - pt1.y);
}

vec3 gradient(vec4& pt_k,
              vec3& pt1,
              vec3& pt2,
              vec4& e,
              mat34& M1,
              mat34& M2){
    vec3 pt1_estimated = M1 * pt_k;
    pt1_estimated /= pt1_estimated[2];
    vec3 pt2_estimated = M2 * pt_k;
    pt2_estimated /= pt2_estimated[2];
    float dis_2 = dis_square(pt1_estimated, pt1) + dis_square(pt2_estimated, pt2);
    float dis = pow(dis_2, -0.5);
    vec3 result;
    result.x = -dis * (e.x * M1(0, 0) + e.y * M1(1, 0) + e.z * M2(0, 0) + e.w * M2(1, 0));
    result.y = -dis * (e.x * M1(0, 1) + e.y * M1(1, 1) + e.z * M2(0, 1) + e.w * M2(1, 1));
    result.z = -dis * (e.x * M1(0, 2) + e.y * M1(1, 2) + e.z * M2(0, 2) + e.w * M2(1, 2));
    return result;
}

vec3 get3dpoint(const vec3& point1,
                const vec3& point2,
                Matrix<double>& M1,
                Matrix<double>& M2,
                bool is_linear = true) {

    Matrix<double> A(4, 4, 0.0);
    A.set_row(point1[0] * M1.get_row(2) - M1.get_row(0), 0);
    A.set_row(point1[1] * M1.get_row(2) - M1.get_row(1), 1);
    A.set_row(point2[0] * M2.get_row(2) - M2.get_row(0), 2);
    A.set_row(point2[1] * M2.get_row(2) - M2.get_row(1), 3);
    Matrix<double> U(4, 4, 0.0);   // initialized with 0s
    Matrix<double> S(4, 4, 0.0);   // initialized with 0s
    Matrix<double> V(4, 4, 0.0);
    svd_decompose(A, U, S, V);
    std::vector<double> pt0 = V.get_column(3);

    vec3 pt;
    vec3 pt_tmp;
    for (int i = 0; i < 3; i++) {
        pt[i] = pt0[i] / pt0[3];
        pt_tmp[i] = pt0[i] / pt0[3];
    }

    if(!is_linear){
        // stop signal for ending loop
        double tolerance = 1;
        int k_max;

        // two observed pt in image planes
        vec3 pt1(point1.x, point1.y, point1.z);
        vec3 pt2(point2.x, point2.y, point2.z);

        // initial pt_k, w value must be 1
        vec4 pt_k(pt0[0] / pt0[3], pt0[1] / pt0[3], pt0[2] / pt0[3], pt0[3] / pt0[3]);

        // initial two estimated pts, w value must be 1
        mat34 mat_M1;
        mat34 mat_M2;
        for (int i = 0; i < 3; ++i){
            mat_M1.set_row(i, vec4 (float(M1.get(i, 0)), float(M1.get(i, 1)), float(M1.get(i, 2)), float(M1.get(i, 3))));
            mat_M2.set_row(i, vec4 (float(M2.get(i, 0)), float(M2.get(i, 1)), float(M2.get(i, 2)), float(M2.get(i, 3))));
        }
//        std::cout << "mat_M1: " << mat_M1 << std::endl;
//        std::cout << "mat_M2: " << mat_M2 << std::endl;

        vec3 pt1_estimated(mat_M1 * pt_k);
        pt1_estimated /= pt1_estimated.z;
        vec3 pt2_estimated(mat_M2 * pt_k);
        pt2_estimated /= pt2_estimated.z;

        // initial matrix J
        std::vector<float> vec_J{-mat_M1(0, 0), -mat_M1(0, 1), -mat_M1(0, 2),
                                 -mat_M1(1, 0), -mat_M1(1, 1), -mat_M1(1, 2),
                                 -mat_M2(0, 0), -mat_M2(0, 1), -mat_M2(0, 2),
                                 -mat_M2(1, 0), -mat_M2(1, 1), -mat_M2(1, 2)};
        mat43 mat_J;
        for (int i = 0; i < 4; ++i){
            vec3 vec_J_row{vec_J[3 * i], vec_J[3 * i + 1], vec_J[3 * i + 2]};
            mat_J.set_row(i, vec_J_row);
        }

        mat3 mat_JT_J(transpose(mat_J) * mat_J);
        mat3 mat_inver_JT_J(inverse(mat_JT_J));

//        std::cout << "mat J: " << mat_J << std::endl;

        float error = dis_square(pt1_estimated, pt1) + dis_square(pt2_estimated, pt2);
        float dis_min = error;
        vec3 best_pt(pt.x, pt.y, pt.z);
        if(error < 1) k_max = 20000;
        else if (error < 2) k_max = 50000;
        else if (error < 5) k_max = 120000;
        else if (error < 10) k_max = 300000;
        else k_max = 500000;

        for (int k = 0; k < k_max && error > tolerance; ++k){
            // refresh vec_c
//            vec4 vec_e{float(pt1.x - (mat_M1(0, 0) * pt_k.x + mat_M1(0, 1) * pt_k.y + mat_M1(0, 2) * pt_k.z + mat_M1(0, 3))),
//                       float(pt1.y - (mat_M1(1, 0) * pt_k.x + mat_M1(1, 1) * pt_k.y + mat_M1(1, 2) * pt_k.z + mat_M1(1, 3))),
//                       float(pt2.x - (mat_M2(0, 0) * pt_k.x + mat_M2(0, 1) * pt_k.y + mat_M2(0, 2) * pt_k.z + mat_M2(0, 3))),
//                       float(pt2.y - (mat_M2(1, 0) * pt_k.x + mat_M2(1, 1) * pt_k.y + mat_M2(1, 2) * pt_k.z + mat_M2(1, 3)))};
            vec4 vec_e{float(pt1.x - pt1_estimated.x),
                       float(pt1.y - pt1_estimated.y),
                       float(pt2.x - pt2_estimated.x),
                       float(pt2.y - pt2_estimated.y)};
//            std::cout << "vec e: " << vec_e << std::endl;

            auto err_P = - mat_inver_JT_J * transpose(mat_J) * to_matrix(vec_e);
//            std::cout << "err P: " << err_P << std::endl;

            // refresh pt_k
//            vec3 gd = gradient(pt_k, pt1, pt2, vec_e, mat_M1, mat_M2);
            pt_k.x += err_P(0, 0);
            pt_k.y += err_P(1, 0);
            pt_k.z += err_P(2, 0);
//            pt_k /= pt_k[3];

            // refresh two estimated pts, w value must be 1
            pt1_estimated = mat_M1 * pt_k;
            pt1_estimated /= pt1_estimated.z;
            pt2_estimated = mat_M2 * pt_k;
            pt2_estimated /= pt2_estimated.z;

            // refresh error
            error = dis_square(pt1_estimated, pt1) + dis_square(pt2_estimated, pt2);
            if (error < dis_min){
                dis_min = error;
                best_pt.x = pt_k.x;
                best_pt.y = pt_k.y;
                best_pt.z = pt_k.z;
//                std::cout << "error: " << error << std::endl;
            }

//            std::cout << "error: " << error << std::endl;
//            std::cout << "pt1: " << pt1 << std::endl;
//            std::cout << "est pt1: " << pt1_estimated << std::endl;
//            std::cout << "pt2: " << pt2 << std::endl;
//            std::cout << "est.pt2: " << pt2_estimated << std::endl;
//            std::cout << "diff(pt1 - pt(k)): x: " << pt1.x - pt1_estimated.x << " y: " << pt1.y - pt1_estimated.y << " z: " << pt1.z - pt1_estimated.z << std::endl;
//            std::cout << "diff(pt2 - pt(k)): x: " << pt2.x - pt2_estimated.x << " y: " << pt2.y - pt2_estimated.y << " z: " << pt2.z - pt2_estimated.z << std::endl;
//            std::cout << "diff(pt3D(1) - pt3D(k)): x: " << pt_tmp.x - pt_k.x << " y: " << pt_tmp.y - pt_k.y << " z: " << pt_tmp.z - pt_k.z << std::endl;
        }

//        if (error <= tolerance) std::cout << "pt error has converged" <<std::endl;
        pt.x = best_pt.x;
        pt.y = best_pt.y;
        pt.z = best_pt.z;
//        std::cout << "pt" << pt << std::endl;
//        std::cout << "pt_tmp" << pt_tmp << std::endl;
    }

    return pt;
}

int countInfront(const std::vector<vec3>& points1,
                 const std::vector<vec3>& points2,
                 Matrix<double>& K,
                 Matrix<double>& R,
                 std::vector<double>& t,
                 bool is_linear = true) {
    int sum = 0;

    Matrix<double> M1(3, 4, 0.0);
    Matrix<double> M2(3, 4, 0.0);
    getM(K, R, t, M1, M2);
    std::cout << "M1" << M1 << std::endl;
    std::cout << "M2" << M2 << std::endl;

    for (int i = 0; i < points1.size(); i++) {
        vec3 P, Q;
        P = get3dpoint(points1[i], points2[i], M1, M2, is_linear);
        mat3 r = to_mat3(R);
        vec3 t0;
        t0[0] = t[0];
        t0[1] = t[1];
        t0[2] = t[2];
        Q = r * P + t0;
        if (P[2] > 0 && Q[2] > 0) {
            sum++;
        }
    }

    return sum;
}

std::vector<vec3> normalize(const std::vector<vec3>& points, mat3& t) {
    double sum_x = 0;
    double sum_y = 0;
    for (auto point : points) {
        sum_x += point[0];
        sum_y += point[1];
    }
    double t_x = sum_x / points.size();
    double t_y = sum_y / points.size();
    std::cout << t_x << std::endl;

    double sum_dist = 0;
    for (auto point : points) {
        sum_dist += sqrt((t_x - point[0]) * (t_x - point[0]) + (t_y - point[1]) * (t_y - point[1]));
    }
    double scale = sqrt(2) / ((sum_dist) / points.size());

    mat3 T;
    T[0] = scale;
    T[1] = 0;
    T[2] = 0;
    T[3] = 0;
    T[4] = scale;
    T[5] = 0;
    T[6] = -t_x * scale;
    T[7] = -t_y * scale;
    T[8] = 1;
    std::vector<vec3> newpoints(points.size());
    t = T;
    //     ┌ scale          0               0 ┐
    // T = | 0              scale           0 |
    //     └ -t_x * scale   -t_y * scale    1 ┘

    for (int i = 0; i < points.size(); i++) {
        newpoints[i] = T * points[i];
    }
    return newpoints;
}

/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'.
 */
bool Triangulation::triangulation(
        float fx, float fy,     /// input: the focal lengths (same for both cameras)
        float cx, float cy,     /// input: the principal point (same for both cameras)
        const std::vector<vec3> &points_0,    /// input: image points (in homogenous coordinates) in the 1st image.
        const std::vector<vec3> &points_1,    /// input: image points (in homogenous coordinates) in the 2nd image.
        std::vector<vec3> &points_3d,         /// output: reconstructed 3D points
        mat3 &R,   /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
        vec3 &t    /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
) const
{
    /// NOTE: there might be multiple workflows for reconstructing 3D geometry from corresponding image points.
    ///       This assignment uses the commonly used one explained in our lecture.
    ///       It is advised to define a function for each sub-task. This way you have a clean and well-structured
    ///       implementation, which also makes testing and debugging easier. You can put your other functions above
    ///       triangulation(), or feel free to put them in one or multiple separate files.

    std::cout << "\nTODO: I am going to implement the triangulation() function in the following file:" << std::endl
              << "\t    - triangulation_method.cpp\n\n";

    std::cout << "[Liangliang]:\n"
                 "\tFeel free to use any data structure and function offered by Easy3D, in particular the following two\n"
                 "\tfiles for vectors and matrices:\n"
                 "\t    - easy3d/core/mat.h  Fixed-size matrices and related functions.\n"
                 "\t    - easy3d/core/vec.h  Fixed-size vectors and related functions.\n"
                 "\tFor matrices with unknown sizes (e.g., when handling an unknown number of corresponding points\n"
                 "\tstored in a file, where their sizes can only be known at run time), a dynamic-sized matrix data\n"
                 "\tstructure is necessary. In this case, you can use the templated 'Matrix' class defined in\n"
                 "\t    - Triangulation/matrix.h  Matrices of arbitrary dimensions and related functions.\n"
                 "\tPlease refer to the corresponding header files for more details of these data structures.\n\n"
                 "\tIf you choose to implement the non-linear method for triangulation (optional task). Please refer to\n"
                 "\t'Tutorial_NonlinearLeastSquares/main.cpp' for an example and some explanations. \n\n"
                 "\tIn your final submission, please\n"
                 "\t    - delete ALL unrelated test or debug code and avoid unnecessary output.\n"
                 "\t    - include all the source code (original code framework + your implementation).\n"
                 "\t    - do NOT include the 'build' directory (which contains the intermediate files in a build step).\n"
                 "\t    - make sure your code compiles and can reproduce your results without any modification.\n\n" << std::flush;

    //--------------------------------------------------------------------------------------------------------------
    // step 1: Check if the input is valid

    // step 1.1: Points in point list must be unique
    // eliminate of duplication points in points_0
    std::set<vec3>set_0(points_0.begin(), points_0.end());
    std::vector<vec3> points_0_unique;
    points_0_unique.assign(set_0.begin(), set_0.end());

    // eliminate of duplication points in points_0
    std::set<vec3>set_1(points_1.begin(), points_1.end());
    std::vector<vec3> points_1_unique;
    points_1_unique.assign(set_1.begin(), set_1.end());

    if (points_0.size() != points_0_unique.size() || points_1.size() != points_1_unique.size()) return false;

    //--------------------------------------------------------------------------------------------------------------
    // step 1.2: Length of two point list must be the same and larger than 8
    if (points_0.size() != points_1.size() && points_0.size() < 8 && points_1.size() < 8) return false;

    //--------------------------------------------------------------------------------------------------------------
    // step 1.3: The w value of points in point list can not be 1
    for (auto pt: points_0){
        if (pt[2] == 0) return false;
    }

    for (auto pt: points_1){
        if (pt[2] != 1) return false;
    }

    //--------------------------------------------------------------------------------------------------------------
    // step 2: Estimate the fundamental matrix F

    // step 2.1: change w values as 1 for each point vector
    for (auto p : points_0) {
        p /= p.z;
    }
    for (auto p : points_1) {
        p /= p.z;
    }

    //--------------------------------------------------------------------------------------------------------------
    // step 2.2: Normaliza each point vector
    mat3 t0;
    auto points_0_normalized = normalize(points_0, t0);
    mat3 t1;
    auto points_1_normalized = normalize(points_1, t1);
    std::cout << "T0:" << t0 << std::endl;
    std::cout << "T0:" << t1 << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 2.3: Generate matrix W
    std::vector<double> vec_W;
    for (int i = 0; i < points_0_normalized.size(); ++i){
        auto p0 = points_0_normalized[i];
        auto p1 = points_1_normalized[i];
        vec_W.push_back(p0.x * p1.x); // W[i][0]
        vec_W.push_back(p0.y * p1.x); // W[i][1]
        vec_W.push_back(p1.x);        // W[i][2]
        vec_W.push_back(p0.x * p1.y); // W[i][3]
        vec_W.push_back(p0.y * p1.y); // W[i][0]
        vec_W.push_back(p1.y);        // W[i][5]
        vec_W.push_back(p0.x);        // W[i][6]
        vec_W.push_back(p0.y);        // W[i][7]
        vec_W.push_back(1);           // W[i][8]
    }
    Matrix<double> mat_W(points_0_normalized.size(), 9, vec_W);

    //--------------------------------------------------------------------------------------------------------------
    // step 2.4: Use SVD to get fundamental matrix F
    // Get original Fq by using SVD(W) to solve W * f = 0
    Matrix<double> mat_U_W(points_0.size(), points_0.size(), 0.0);   // M * M matrix, initialized with 0s
    Matrix<double> mat_S_W(points_0.size(), 9, 0.0);           // M * N matrix, initialized with 0s
    Matrix<double> mat_V_W(9, 9, 0.0);                   // N * N matrix, initialized with 0s
    svd_decompose(mat_W, mat_U_W, mat_S_W, mat_V_W);
    auto vec_F_original = mat_V_W.get_column(8);
    Matrix<double> mat_F_original(3, 3, vec_F_original);
    std::cout << "original matrix F:" << mat_F_original << std::endl;

    // Get estimated Fq by using SVD(F)
    Matrix<double> mat_U_F(3, 3, 0.0);   // M * M matrix, initialized with 0s
    Matrix<double> mat_S_F(3, 3, 0.0);   // M * M matrix, initialized with 0s
    Matrix<double> mat_V_F(3, 3, 0.0);   // M * M matrix, initialized with 0s
    svd_decompose(mat_F_original, mat_U_F , mat_S_F, mat_V_F);
    Matrix<double> mat_S_F_estimated(mat_S_F);
    mat_S_F_estimated.set(2, 2, 0);
    Matrix<double> mat_F_estimated(mat_U_F * mat_S_F_estimated * mat_V_F.transpose());
    std::cout << "estimated matrix F:" << mat_F_estimated << std::endl;

    // Denormalization
    mat3 mat3_F(transpose(t1) * to_mat3(mat_F_estimated) * t0);
    Matrix<double> mat_F(to_Matrix(mat3_F));
    std::cout << "final matrix F:" << mat_F << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 3: Compute the essential matrix E

    // step 3.1: Generate the intrinsic parameter matrix K
    Matrix<double> mat_K(3, 3, std::vector<double>{fx, 0, cx, 0, fy, cy, 0, 0, 1});

    //--------------------------------------------------------------------------------------------------------------
    // step 3.2: Compute the essential matrix E
    Matrix<double> mat_E(mat_K.transpose() * mat_F * mat_K);
    std::cout << "matrix E:" << mat_E << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 4: Recover rotation R and t

    // step 4.1: Recover matrix R
    // Use SVD(E) to decomposition E
    Matrix<double> mat_U_E(3, 3, 0.0);   // M * M matrix, initialized with 0s
    Matrix<double> mat_S_E(3, 3, 0.0);   // M * M matrix, initialized with 0s
    Matrix<double> mat_V_E(3, 3, 0.0);   // M * M matrix, initialized with 0s
    svd_decompose(mat_E, mat_U_E, mat_S_E, mat_V_E);

    // Generate two component matrices in decomposition of matrix E
    Matrix<double> mat_W_component(3, 3, std::vector<double>{0, -1, 0, 1, 0, 0, 0, 0, 1});
    Matrix<double> mat_Z_component(3, 3, std::vector<double>{0, 1, 0, -1, 0, 0, 0, 0, 0});

    // Calculate matrices R candidates
    Matrix<double> mat_R1(determinant(mat_U_E *
                                      mat_W_component *
                                      mat_V_E.transpose()) *
                          mat_U_E *
                          mat_W_component *
                          mat_V_E.transpose());
    Matrix<double> mat_R2(determinant(mat_U_E *
                                      mat_W_component.transpose() *
                                      mat_V_E.transpose()) *
                          mat_U_E *
                          mat_W_component.transpose() *
                          mat_V_E.transpose());
    std::cout << "R1:" << mat_R1 << std::endl;
    std::cout << "R2:" << mat_R2 << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 4.2: Recover vector t
    // Calculate vector t candidates
    std::vector<double> vec_t1 {mat_U_E.get(0, 2),
                                mat_U_E.get(1, 2),
                                mat_U_E.get(2, 2)};
    std::vector<double> vec_t2 {-mat_U_E.get(0, 2),
                                -mat_U_E.get(1, 2),
                                -mat_U_E.get(2, 2)};
    std::cout << "T1:" << vec_t1 << std::endl;
    std::cout << "T2:" << vec_t2 << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 4.3: Find the correct R and t
    std::vector<int> sumVector(4);
    sumVector[0] = countInfront(points_0, points_1, mat_K, mat_R1, vec_t1);
    sumVector[1] = countInfront(points_0, points_1, mat_K, mat_R1, vec_t2);
    sumVector[2] = countInfront(points_0, points_1, mat_K, mat_R2, vec_t1);
    sumVector[3] = countInfront(points_0, points_1, mat_K, mat_R2, vec_t2);
    int max = 0, num = 5;
    for (int i = 0; i < 4; i++) {
        if (sumVector[i] > max) {
            max = sumVector[i];
            num = i;
        }
    }
    switch (num)
    {
        case 0:
            R = to_mat3(mat_R1);
            t[0] = t1[0];
            t[1] = t1[1];
            t[2] = t1[2];
            break;
        case 1:
            R = to_mat3(mat_R1);
            t[0] = vec_t2[0];
            t[1] = vec_t2[1];
            t[2] = vec_t2[2];
            break;
        case 2:
            R = to_mat3(mat_R2);
            t[0] = t1[0];
            t[1] = t1[1];
            t[2] = t1[2];
            break;
        case 3:
            R = to_mat3(mat_R2);
            t[0] = vec_t2[0];
            t[1] = vec_t2[1];
            t[2] = vec_t2[2];
            break;
        default:
            break;
    }

    std::cout << "R:" << R << std::endl;
    std::cout << "t:" << t << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 5: Triangulation

    // step 5.1: Compute the projection matrix from K,R,t
    Matrix<double> mat_M1, mat_M2;
    std::vector<double> tt(3);
    tt[0] = t[0];
    tt[1] = t[1];
    tt[2] = t[2];
    Matrix<double> mat_R(to_Matrix(R));
    getM(mat_K, mat_R, tt, mat_M1, mat_M2);
    std::cout << "M1" << mat_M1 << std::endl;
    std::cout << "M2" << mat_M2 << std::endl;

    //--------------------------------------------------------------------------------------------------------------
    // step 5.2: Compute the 3D points using linear method
    for (int i = 0; i < points_0.size(); i++) {
        vec3 pt = get3dpoint(points_0[i], points_1[i], mat_M1, mat_M2, false);
        points_3d.push_back(pt);
    }

    return !points_3d.empty();
}
