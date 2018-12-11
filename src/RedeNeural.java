import java.util.Random;

public class RedeNeural{
    private int quantidadeEntrada;
    private int neuroniosEscondidos;
    private int tamanhoSaida;
    private double taxaDeAprendizagem;
    private int epocas;
    private double pesosOcultos[][];
    private double pesosSaida[][];
    private double entrada[];
    private double layer[];
    private double alvo[];
    private double vetor[];
    private double erroSaida[];
    private double erroCamadas[];

    private int[][] train;
    private int[][] trainOutput;
    public RedeNeural(int quantidadeEntrada, int neuroniosEscondidos, int tamanhoSaida, double taxaDeAprendizagem, int epocas) {
        this.quantidadeEntrada = quantidadeEntrada;
        this.neuroniosEscondidos = neuroniosEscondidos;
        this.tamanhoSaida = tamanhoSaida;
        this.taxaDeAprendizagem = taxaDeAprendizagem;
        this.epocas = epocas;
        this.pesosOcultos = new double[quantidadeEntrada+1][neuroniosEscondidos];
        this.pesosSaida = new double[neuroniosEscondidos+1][tamanhoSaida];
        this.entrada = new double[quantidadeEntrada];
        this.layer = new double[neuroniosEscondidos];
        this.alvo = new double[tamanhoSaida];
        this.vetor = new double[tamanhoSaida];
        this.erroCamadas = new double[neuroniosEscondidos];
        this.erroSaida = new double[tamanhoSaida];


    }
    public void treinar(int[][] entrada, int[][] saida){
        this.train = entrada;
        this.trainOutput = saida;


        setaPesos();
        for(int epoca = 0; epoca < this.epocas; epoca++){
            for(int entradas = 0; entradas < entrada.length; entradas++){

                for (int i = 0; i < this.quantidadeEntrada; i++) {
                    this.entrada[i] = this.train[entradas][i];
                }

                for (int i = 0; i < this.tamanhoSaida; i++) {
                    this.alvo[i]= this.trainOutput[entradas][i];
                }

                feedForward();

                backPropagation();

            }

        }
        getTraining();
        System.out.println("Testando a rede");
        test(this.train);

    }
    private void getTraining() {
        double soma = 0.0;
        for (int i = 0; i < this.train.length; i++) {
            for (int j = 0; j < this.quantidadeEntrada; j++) {
                this.entrada[j] = this.train[i][j];
            }

            for (int j = 0; j < this.tamanhoSaida; j++) {
                this.alvo[j] = trainOutput[i][j];
            }

            feedForward();

            if (maximo(this.vetor) == maximo(this.alvo)) {
                soma += 1;
            } else {
                System.out.println(maximo(this.vetor) + "\t" + maximo(this.alvo));
            }
        }




    }

    private  void test(int[][] testRede) {
        for (int i = 0; i < testRede.length; i++) {
            for (int j = 0; j < this.quantidadeEntrada; j++) {
                this.entrada[j] = testRede[i][j];
            }

            feedForward();

            for (int j = 0; j < this.quantidadeEntrada; j++) {
                System.out.print(this.entrada[j] + "\t");
            }

            System.out.print("Output: " + this.vetor[maximo(this.vetor)] +"\n");
        }
    }
    private  int maximo(double[] vetor) {

        int sel = 0;
        double max = vetor[sel];

        for (int i = 0; i < this.tamanhoSaida; i++) {
            if (vetor[i] >= max) {
                max = vetor[i];
                sel = i;
            }
        }
        return sel;
    }

    private void backPropagation(){


            for (int i = 0; i < this.tamanhoSaida; i++) {
                this.erroSaida[i] = (this.alvo[i] - this.vetor[i]) * sigmoidDerivada(this.vetor[i]);
            }

            for (int j = 0; j < this.neuroniosEscondidos; j++) {
                this.erroCamadas[j] = 0.0;
                for (int i = 0; i < this.tamanhoSaida; i++) {
                    this.erroCamadas[j] += this.erroSaida[i] * this.pesosSaida[j][i];
                }
                this.erroCamadas[j] *= sigmoidDerivada(this.layer[j]);
            }

            for (int out = 0; out < this.tamanhoSaida; out++) {
                for (int hid = 0; hid < this.neuroniosEscondidos; hid++) {
                    this.pesosSaida[hid][out] += (this.taxaDeAprendizagem * this.erroSaida[out] * this.layer[hid]);
                }
                this.pesosSaida[this.neuroniosEscondidos][out] += (this.taxaDeAprendizagem * this.erroSaida[out]);
            }

            for (int i = 0; i < this.neuroniosEscondidos; i++) {
                for (int k = 0; k < this.quantidadeEntrada; k++) {
                    this.pesosOcultos[k][i] += (this.taxaDeAprendizagem * this.erroCamadas[i] * this.entrada[k]);
                }
                this.pesosOcultos[this.quantidadeEntrada][i] += (this.taxaDeAprendizagem * this.erroCamadas[i]);
            }




    }
    private void feedForward() {

        double soma = 0.0;


        for (int i = 0; i < this.neuroniosEscondidos; i++) {
            soma = 0.0;
            for (int j = 0; j < this.quantidadeEntrada; j++) {
                soma += this.entrada[j] * this.pesosOcultos[j][i];
            }

            soma += this.pesosOcultos[this.quantidadeEntrada][i];
            this.layer[i] = sigmoid(soma);
        }

        for (int i = 0; i < this.tamanhoSaida; i++) {
            soma = 0.0;
            for (int j = 0; j < this.neuroniosEscondidos; j++) {
                soma += this.layer[j] * this.pesosSaida[j][i];
            }

            soma += this.pesosSaida[this.neuroniosEscondidos][i];
            this.vetor[i] = sigmoid(soma);
        }

    }
    private double sigmoid(double valor){
        return (1.0 / (1.0 + Math.exp(-valor)));
    }
    private static double sigmoidDerivada( double valor) {
        return (valor * (1.0 - valor));
    }
    private void setaPesos(){
        for (int i = 0; i <= this.quantidadeEntrada; i++)
        {
            for (int j = 0; j < this.neuroniosEscondidos; j++) {
                this.pesosOcultos[i][j] = new Random().nextDouble() - 0.5;
            }
        }

        for (int i = 0; i <= this.neuroniosEscondidos; i++)
        {
            for (int j = 0; j < this.tamanhoSaida; j++) {

                this.pesosSaida[i][j] = new Random().nextDouble() - 0.5;
            }
        }


    }
}