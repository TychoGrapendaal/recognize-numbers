import java.util.Arrays;
import java.util.Scanner;

public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] data;


    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public void normalize() {
        double sum = 0;
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] *= (1.0/sum);
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (double[] datum : data) {
            s.append(Arrays.toString(datum));
            s.append("\n");
        }

        return s.toString();
    }



    public void set(int i, int j, double n) {
        data[i][j] = n;
    }

    public static Matrix readMatrix(Scanner scanner, int r, int c) {
        scanner.useDelimiter(", |\n");
        Matrix m = new Matrix(r,c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                m.data[i][j] = Double.parseDouble(scanner.next().replace("[", "").replace("]",""));
            }
        }
        return m;
    }

    public void scale(double n) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] *= n;
            }
        }
    }

    public static Matrix dotProduct(Matrix a, Matrix b) throws Exception {
        if (a.cols != b.rows) {
            throw new Exception("The rows of a do not equal the cols of b");
        }
        Matrix result = new Matrix(a.rows, b.cols);
        for (int i = 0; i < result.rows; i ++) {
            for (int j = 0; j < result.cols; j++) {
                double sum = 0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i,j,sum);
            }
        }
        return result;
    }

    public void add(double n) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] += n;
            }
        }
    }

    public void addMatrix(Matrix m) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] += m.data[i][j];
            }
        }
    }

    public void subtractMatrix(Matrix m) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] -= m.get(i,j);
            }
        }
    }

    public static Matrix subtract(Matrix a, Matrix b) throws Exception {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new Exception("The matrices are of different sizes (subtract)");
        }
        Matrix r = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                r.set(i,j, a.get(i,j) - b.get(i,j));
            }
        }
        return r;
    }

    public void randomize() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] = Math.random()*2-1;
            }
        }
    }

    public double get(int i, int j) {
        return data[i][j];
    }

    public static Matrix transpose(Matrix m) {
        Matrix r = new Matrix(m.cols, m.rows);
        for (int i = 0; i < r.rows; i++) {
            for (int j = 0; j < r.cols; j++) {
                r.set(i,j,m.get(j,i));
            }
        }
        return r;
    }

    public void print() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                System.out.print(this.get(i,j) + "|");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static Matrix fromArray(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.set(i,0,arr[i]);
        }
        return m;
    }

    public void sigmoid() {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                data[i][j] = 1 / (1 + Math.exp(-data[i][j]));
            }
        }
    }

    public static Matrix staticSigmoid(Matrix m) {
        Matrix r = new Matrix(m.rows, m.cols);
        for (int i = 0; i < r.rows; i++) {
            for (int j = 0; j < r.cols; j++) {
                r.data[i][j] = 1 / (1 + Math.exp(-m.data[i][j]));
            }
        }
        return r;
    }

    public static Matrix error(Matrix m) {
        Matrix r = new Matrix(m.rows, m.cols);
        for (int i = 0; i < r.rows; i++) {
            for (int j = 0; j < r.cols; j++) {
                r.data[i][j] = m.data[i][j] * (1 - m.data[i][j]);
            }
        }
        return r;
    }

    public void multiply(Matrix m) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] *= m.data[i][j];
            }
        }
    }

    public void mutate(double p) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                if (Math.random() < p) {
                    this.set(i,j,Math.random() * 2 - 1);
                }
            }
        }
    }

    public Matrix copy() {
        Matrix m = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            if (this.cols >= 0) System.arraycopy(this.data[i], 0, m.data[i], 0, this.cols);
        }
        return m;
    }

    public double[] toArray() {
        double[] arr = new double[this.rows*this.cols];
        for (int i = 0; i < this.rows; i++) {
            if (this.cols >= 0) System.arraycopy(this.data[i], 0, arr, i * this.cols, this.cols);
        }
        return arr;
    }


}