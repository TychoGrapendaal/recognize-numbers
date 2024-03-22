import java.util.Scanner;

public class NeuralNetwork {
    private final int input_nodes;
    private final int numberOfHid;
    private final int hidden_nodes;
    private final int output_nodes;
    private double learning_rate;

    private Matrix[] weights;
    private Matrix[] bias;
    
    public NeuralNetwork(NeuralNetwork a) {
        this.input_nodes = a.input_nodes;
        this.numberOfHid = a.numberOfHid;
        this.hidden_nodes = a.hidden_nodes;
        this.output_nodes = a.output_nodes;
        this.weights = new Matrix[a.numberOfHid+1];
        this.bias = new Matrix[numberOfHid+1];

        this.weights[0] = a.weights[0].copy();
        for (int i = 1; i < a.numberOfHid; i++) {
            this.weights[i] = a.weights[i];
            this.bias[i] = a.bias[i].copy();
        }
        this.weights[a.numberOfHid] = a.weights[a.numberOfHid].copy();

        this.bias[a.numberOfHid] = a.bias[a.numberOfHid].copy();
        this.learning_rate = a.learning_rate;
    }

    public NeuralNetwork(int in_nodes, int numberOfHid, int hid_nodes, int out_nodes) {
        this.input_nodes = in_nodes;
        this.numberOfHid = numberOfHid;
        this.hidden_nodes = hid_nodes;
        this.output_nodes = out_nodes;

        this.weights = new Matrix[numberOfHid+1];
        this.bias = new Matrix[numberOfHid+1];

        this.weights[0] = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights[0].randomize();
        this.bias[0] = new Matrix(this.hidden_nodes,1);
        this.bias[0].randomize();
        for (int i = 1; i < numberOfHid; i++) {
            this.weights[i] = new Matrix(this.hidden_nodes, this.hidden_nodes);
            this.weights[i].randomize();
            this.bias[i] = new Matrix(this.hidden_nodes, 1);
            this.bias[i].randomize();
        }
        this.weights[numberOfHid] = new Matrix(this.output_nodes, this.hidden_nodes);
        this.weights[numberOfHid].randomize();


        this.bias[numberOfHid] = new Matrix(this.output_nodes, 1);
        this.bias[numberOfHid].randomize();

        this.setLearningRate(0.1);
    }

    @Override
    public String toString() {
        StringBuilder weights = new StringBuilder();
        StringBuilder bias = new StringBuilder();

        for (Matrix m : this.weights) {
            weights.append(m.toString());
            weights.append("\n");
        }
        for (Matrix m : this.bias) {
            bias.append(m.toString());
            bias.append("\n");
        }
        return
                input_nodes +
                "\n" + numberOfHid +
                "\n" + hidden_nodes +
                "\n"+ output_nodes +
                "\n"+ weights +
                "\n"+ bias;
    }

    public static NeuralNetwork readNN(Scanner scanner) {
        String n = scanner.nextLine();
        int in = Integer.parseInt(n);
        int numOfHid = Integer.parseInt(scanner.nextLine());
        int hid = Integer.parseInt(scanner.nextLine());
        int out = Integer.parseInt(scanner.nextLine());
        Matrix[] weights = new Matrix[numOfHid+1];
        Matrix[] bias = new Matrix[numOfHid+1];
        weights[0] = Matrix.readMatrix(scanner, hid, in);
        scanner.nextLine();

        for (int i = 1; i < numOfHid; i++) {
            weights[i] = Matrix.readMatrix(scanner, hid, hid);
            scanner.nextLine();
        }
        weights[numOfHid] = Matrix.readMatrix(scanner, out, hid);
        scanner.nextLine();
        scanner.nextLine();

        for (int i = 0; i < numOfHid; i++) {
            bias[i] = Matrix.readMatrix(scanner, hid, 1);
            scanner.nextLine();
        }

        bias[numOfHid] = Matrix.readMatrix(scanner, out, 1);

        NeuralNetwork nn = new NeuralNetwork(in,numOfHid,hid,out);
        nn.weights = weights;
        nn.bias = bias;
        return nn;
    }


    public double[] predict(double[] input_array) throws Exception {

        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(input_array);
        Matrix[] hidden = new Matrix[this.numberOfHid];
        hidden[0] = Matrix.dotProduct(this.weights[0], inputs);
        hidden[0].addMatrix(this.bias[0]);
        // activation function!
        hidden[0].sigmoid();

        for (int i = 1; i < this.numberOfHid; i++) {
            hidden[i] = Matrix.dotProduct(this.weights[i], hidden[i-1]);
            hidden[i].addMatrix(this.bias[i]);
            hidden[i].sigmoid();
        }

        // Generating the output's output!
        Matrix output = Matrix.dotProduct(this.weights[this.numberOfHid], hidden[this.numberOfHid-1]);
        output.addMatrix(this.bias[this.numberOfHid]);
        output.sigmoid();

        // Sending back to the caller!
        return output.toArray();
    }

    public void setLearningRate(double learning_rate) {
        this.learning_rate = learning_rate;
    }


    public void train(double[] input_array, double[] target_array) throws Exception {
        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(input_array);
        Matrix[] hidden = new Matrix[this.numberOfHid];
        hidden[0] = Matrix.dotProduct(this.weights[0], inputs);
        hidden[0].addMatrix(this.bias[0]);
        // activation function!
        hidden[0].sigmoid();

        for (int i = 1; i < this.numberOfHid; i++) {
            hidden[i] = Matrix.dotProduct(this.weights[i], hidden[i-1]);
            hidden[i].addMatrix(this.bias[i]);
            hidden[i].sigmoid();
        }

        // Generating the output's output!
        Matrix outputs = Matrix.dotProduct(this.weights[this.numberOfHid], hidden[this.numberOfHid-1]);
        outputs.addMatrix(this.bias[this.numberOfHid]);
        outputs.sigmoid();

        // Convert array to matrix object
        Matrix targets = Matrix.fromArray(target_array);

        // Calculate the error
        //  = TARGETS - OUTPUTS
        Matrix output_errors = Matrix.subtract(targets, outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        Matrix gradients = Matrix.error(outputs);
        gradients.multiply(output_errors);
        gradients.scale(this.learning_rate);


        // Calculate deltas
        Matrix hidden_T = Matrix.transpose(hidden[this.numberOfHid-1]);
        Matrix weight_ho_deltas = Matrix.dotProduct(gradients, hidden_T);

        // Adjust the weights by deltas
        this.weights[this.numberOfHid].addMatrix(weight_ho_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias[this.numberOfHid].addMatrix(gradients);

        for (int i = this.numberOfHid-1; i > 0; i--) {

            // Calculate the hidden layer errors
            Matrix who_t = Matrix.transpose(this.weights[i+1]);
            output_errors = Matrix.dotProduct(who_t, output_errors);

            // Calculate hidden gradient
            Matrix hidden_gradient = Matrix.error(hidden[i]);
            hidden_gradient.multiply(output_errors);
            hidden_gradient.scale(this.learning_rate);

            // Calculate input->hidden deltas
            Matrix inputs_T = Matrix.transpose(inputs);
            Matrix weight_ih_deltas = Matrix.dotProduct(hidden_gradient, inputs_T);

            this.weights[i].addMatrix(weight_ih_deltas);
            // Adjust the bias by its deltas (which is just the gradients)
            this.bias[i].addMatrix(hidden_gradient);

        }

        // Calculate the hidden layer errors
        Matrix who_t = Matrix.transpose(this.weights[1]);
        Matrix hidden_errors = Matrix.dotProduct(who_t, output_errors);

        // Calculate hidden gradient
        Matrix hidden_gradient = Matrix.error(hidden[0]);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.scale(this.learning_rate);

        // Calculate input->hidden deltas
        Matrix inputs_T = Matrix.transpose(inputs);
        Matrix weight_ih_deltas = Matrix.dotProduct(hidden_gradient, inputs_T);

        this.weights[0].addMatrix(weight_ih_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias[0].addMatrix(hidden_gradient);
    }

    public NeuralNetwork copy() {
        return new NeuralNetwork(this);
    }

    public void mutate(double p) {
        for (Matrix weight : this.weights) {
            weight.mutate(p);
        }
        for (Matrix bias : this.bias) {
            bias.mutate(p);
        }
    }
}
