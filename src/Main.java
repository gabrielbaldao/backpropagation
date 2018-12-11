public class Main {
    public static void main(String[] args) {

        /*Configuracao da Rede*/
        int quantidadeDeEntrada = 2;
        int neuroniosEscondidos = 12;
        int quantidadeSaida = 1;

        double taxaDeAprendizagem = 0.1;
        int epocas = 10000;
         int entrada[][] = new int[][]
                {{ 0, 0},{0, 1},{1, 0},{1,1}};

         int saida[][] = new int[][]
              //  {{1},{0},{0},{1}};
                 {{0},{1},{1},{0}};



      RedeNeural rn =  new RedeNeural(quantidadeDeEntrada, neuroniosEscondidos,quantidadeSaida,taxaDeAprendizagem,epocas);
      rn.treinar(entrada,saida);

    }
}
