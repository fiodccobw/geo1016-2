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
#include<cmath>


using namespace easy3d;


/// convert a 3 by 3 matrix of type 'Matrix<double>' to mat3
mat3 to_mat3(Matrix<double>& M) {
    mat3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}


/// convert M of type 'matN' (N can be any positive integer) to type 'Matrix<double>'
template<typename mat>
Matrix<double> to_Matrix(const mat& M) {
    const int num_rows = M.num_rows();
    const int num_cols = M.num_columns();
    Matrix<double> result(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

std::vector<vec3> normalize(const std::vector<vec3>& points, mat3& t) {
    //mat3 T;
    double sumx = 0;
    double sumy = 0;
    for (int i = 0; i < points.size(); i++) {
        sumx += points[i][0];
        sumy += points[i][1];
    }
    double tx = sumx / points.size();
    double ty = sumy / points.size();
    double sumdist = 0;
    std::cout << tx << std::endl;
    for (int i = 0; i < points.size(); i++) {
        sumdist += sqrt((tx - points[i][0]) * (tx - points[i][0]) + (ty - points[i][1]) * (ty - points[i][1]));
    }
    double scale = sqrt(2) / ((sumdist) / points.size());

    mat3 T;
    T[0] = scale;
    T[1] = 0;
    T[2] = 0;
    T[3] = 0;
    T[5] = 0;
    T[4] = scale;
    T[6] = -tx * scale;
    T[7] = -ty * scale;
    T[8] = 1;
    std::vector<vec3> newpoints(points.size());
    t = T;

    for (int i = 0; i < points.size(); i++) {
        newpoints[i] = T * points[i];
    }


    return newpoints;
}



void getM(Matrix<double>& K, Matrix<double>& R, std::vector<double>& t, Matrix<double>& M1, Matrix<double>& M2) {
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

vec3 get3dpoint(const vec3& point1, const vec3& point2, Matrix<double>& M1, Matrix<double>& M2) {
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
    for (int i = 0; i < 3; i++) {
        pt[i] = pt0[i] / pt0[3];
    }
    return pt;
}

int countInfront(const std::vector<vec3>& points1, const std::vector<vec3>& points2, Matrix<double>& K, Matrix<double>& R, std::vector<double>& t) {
    int sum = 0;

    Matrix<double> M1(3, 4, 0.0);
    Matrix<double> M2(3, 4, 0.0);
    getM(K, R, t, M1, M2);
    std::cout << "M1" << M1 << std::endl;
    std::cout << "M2" << M2 << std::endl;

    for (int i = 0; i < points1.size(); i++) {
        vec3 P, Q;
        P = get3dpoint(points1[i], points2[i], M1, M2);
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


/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'.
 */
bool Triangulation::triangulation(
    float fx, float fy,     /// input: the focal lengths (same for both cameras)
    float cx, float cy,     /// input: the principal point (same for both cameras)
    const std::vector<vec3>& points_0,    /// input: image points (in homogenous coordinates) in the 1st image.
    const std::vector<vec3>& points_1,    /// input: image points (in homogenous coordinates) in the 2nd image.
    std::vector<vec3>& points_3d,         /// output: reconstructed 3D points
    mat3& R,   /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
    vec3& t    /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
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

    //STEP1--------------------------------------------
    //1.1 Normalization
    mat3 T;
    mat3 T0, T1;
    std::vector<vec3> points_0n = normalize(points_0, T0);
    std::vector<vec3> points_1n = normalize(points_1, T1);
    std::cout << "T:" << T0 << std::endl;
    std::cout << "T:" << T1 << std::endl;

    //construct matrix W
    Matrix<double> W0(points_0.size(), 9, 0.0);
    for (int i = 0; i < points_0.size(); i++) {
        W0[i][0] = points_0n[i][0] * points_1n[i][0];
        W0[i][1] = points_0n[i][1] * points_1n[i][0];
        W0[i][2] = points_1n[i][0];
        W0[i][3] = points_0n[i][0] * points_1n[i][1];
        W0[i][4] = points_0n[i][1] * points_1n[i][1];
        W0[i][5] = points_1n[i][1];
        W0[i][6] = points_0n[i][0];
        W0[i][7] = points_0n[i][1];
        W0[i][8] = 1;
    }

    //1.2 Solve the origin F
    std::cout << "---- first SVD--------- \n" << '\n';
    Matrix<double> U1(points_0.size(), points_0.size(), 0.0);   // initialized with 0s
    Matrix<double> S1(points_0.size(), 9, 0.0);   // initialized with 0s
    Matrix<double> V1(9, 9, 0.0);   // initialized with 0s
    svd_decompose(W0, U1, S1, V1);

    mat3 F;
    for (int i = 0; i < 3; i++) {
        F[i] = V1[i * 3][8];
        F[i + 3] = V1[i * 3 + 1][8];
        F[i + 6] = V1[i * 3 + 2][8];
    }
    Matrix<double> F1 = to_Matrix(F);
    std::cout << "F1:" << F1 << std::endl;


    //1.3 Constraint enforcement
    Matrix<double> UF(3, 3, 0.0);   // initialized with 0s
    Matrix<double> SF(3, 3, 0.0);   // initialized with 0s
    Matrix<double> VF(3, 3, 0.0);   // initialized with 0s
    svd_decompose(F1, UF, SF, VF);
    std::cout << "SF:" << SF << std::endl;
    SF[2][2] = 0;
    F1 = UF * SF * VF.transpose();
    std::cout << "F after enforcement:" << F1 << std::endl;


    //1.4 Denormalization
    mat3 F2 = to_mat3(F1);
    F2 = transpose(T1) * F2 * T0;
    for (int i = 0; i < 9; i++) {
        F2[i] /= F2[8];
    }
    std::cout << "F FINAL:" << F2 << std::endl;
    Matrix<double> _F = to_Matrix(F2);




    //STEP2--------------------------------------
    //2.1 Get essential Matirx E
    Matrix<double> K(3, 3, 0.0);
    K[0][0] = fx;
    K[1][1] = fy;
    K[0][2] = cx;
    K[1][2] = cy;
    K[2][2] = 1;
    std::cout << "K" << K << std::endl;
    Matrix<double> E(3, 3, 0.0);
    E = K.transpose() * _F * K;
    std::cout << "E" << E << std::endl;

    //2.2 Find 4 camera pose matrix
    Matrix<double> U(3, 3, 0.0);
    Matrix<double> S(3, 3, 0.0);
    Matrix<double> V(3, 3, 0.0);

    svd_decompose(E, U, S, V);
    Matrix<double> W(3, 3, 0.0);
    Matrix<double> Z(3, 3, 0.0);

    W[0][1] = -1;
    W[1][0] = 1;
    W[2][2] = 1;
    Z[0][1] = 1;
    Z[1][0] = -1;

    std::cout << "W:" << W << std::endl;
    std::cout << "Z:" << Z << std::endl;

    Matrix<double> R1(3, 3, 0.0);
    Matrix<double> R2(3, 3, 0.0);
    R1 = determinant(U * W * V.transpose()) * U * W * V.transpose();
    R2 = determinant(U * W.transpose() * V.transpose()) * U * W.transpose() * V.transpose();

    std::vector<double> t1(3);
    std::vector<double> t2(3);
    for (int i = 0; i < 3; i++) {
        t1[i] = U[i][2];
        t2[i] = -U[i][2];
    }
    std::cout << "R1:" << R1 << std::endl;
    std::cout << "R2:" << R2 << std::endl;
    std::cout << "T1:" << t1 << std::endl;
    std::cout << "T2:" << t2 << std::endl;

    //2.3 Determine the correct relative pose
    std::vector<int> sumVector(4);
    sumVector[0] = countInfront(points_0, points_1, K, R1, t1);
    sumVector[1] = countInfront(points_0, points_1, K, R1, t2);
    sumVector[2] = countInfront(points_0, points_1, K, R2, t1);
    sumVector[3] = countInfront(points_0, points_1, K, R2, t2);
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
        R = to_mat3(R1);
        t[0] = t1[0];
        t[1] = t1[1];
        t[2] = t1[2];
        break;
    case 1:
        R = to_mat3(R1);
        t[0] = t2[0];
        t[1] = t2[1];
        t[2] = t2[2];
        break;
    case 2:
        R = to_mat3(R2);
        t[0] = t1[0];
        t[1] = t1[1];
        t[2] = t1[2];
        break;
    case 3:
        R = to_mat3(R2);
        t[0] = t2[0];
        t[1] = t2[1];
        t[2] = t2[2];
        break;
    default:
        break;
    }

    std::cout << "R:" << R << std::endl;
    std::cout << "t:" << t << std::endl;


    //STEP3----------------------------------
    //3.1 Compute the projection matrix from K,R,t
    Matrix<double> M1, M2;
    std::vector<double> tt(3);
    tt[0] = t[0];
    tt[1] = t[1];
    tt[2] = t[2];
    getM(K, to_Matrix(R), tt, M1, M2);
    std::cout << "M1" << M1 << std::endl;
    std::cout << "M2" << M2 << std::endl;

    //3.2 Compute the 3D points using linear method 
    for (int i = 0; i < points_0.size(); i++) {
        vec3 pt = get3dpoint(points_0[i], points_1[i], M1, M2);
        points_3d.push_back(pt);
    }

    return points_3d.size() > 0;
}