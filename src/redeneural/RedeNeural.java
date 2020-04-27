package redeneural;

import java.util.*;
import java.lang.*;
import java.lang.Math;


public class RedeNeural {
    
    public static final double VERDADEIRO = 1;
    public static final double FALSO = -1;
    public static final double NEUTRO = 0;
    
    
    public double[] inpA; // Entradas da rede
    public double erroTotal; // Erro total da rede
    public double hidA[]; // Saídas da camada oculta
    public double hidW[][]; // Pesos entre camada de entrada e camada oculta
    
    
    
    public double outA[]; // Saídas da camada de saída (saída ideal)
    public double outW[][]; //Pesos da camada de saída
    public double outD[]; // Erro de cada neuronio da camada de saida
    public double outN[]; // Saída real da rede
    
    
    public int nInp; // Número de entradas (Inputs)
    public int nHid; // Número de neuronios da camada oculta
    public int nOut; // Número de neruonios da camada de saída
    
    
    
    public double eida; // Taxa de aprendizado
    public double teta; // Linear da função sigmoid
    public double elast; // Elasticidade
    
    public RedeNeural(int i, int h, int o, double ei, double th, double el){
        
        nInp = i;
        nHid = h;
        nOut = o;
        eida = ei;
        teta = th;
        elast = el;
        
        this.inpA = new double[i]; // Inicializando o número de entradas
        this.hidA = new double[h]; // Inicializa as saídas da camada oculta
        this.hidW = new double[h][i]; // Inicializa os pesos da camada oculta
        this.outA = new double[o]; // Inicializa  as saídas da camada de saída
        this.outW = new double[o][h]; // Inicializa os pesos da camada de saída
        this.outD = new double[o]; // Inicializa o erro 
        this.outN = new double[o]; // Inicializa a saida real
        
        this.inicia();
        
        
    }
        public void inicia(){
        
        for(int i = 0; i < nInp; i++){
            
            inpA[i] = frandom(-1, 1);
            
        }
        
        for(int i = 0; i < nHid; i++){
            hidA[i] = frandom(-1, 1);
            for(int j = 0; j < nInp; j++){
                hidW[i][j] = frandom(-1, 1);
            
            }
        }
        
        for(int i = 0; i < nOut; i++){
            for(int j = 0; j < nHid; j++){
                outW[i][j] = frandom(-1, 1);
            }
        }
        
        erroTotal = 0;
        
    }
    
    public void propagacao(){
        
        double sum2;
        
        for(int i = 0; i < nHid; i++){
            sum2 = 0;
            for(int j = 0; j < nOut; j++){
                sum2 += hidW[i][j]*inpA[j];
            }
            hidA[i] = tamh(sum2);
        }
        
        for(int i = 0; i < nOut; i++){
            sum2 = 0;
            for(int j = 0; j < nHid; j++){
                sum2 += outW[i][j] * hidA[j];
            }
            outN[i] = tamh(sum2);
        }
        
    }
    
    public double tamh(double x){
        
        return 2*Math.exp(-1*elast*x-teta)/(1 + Math.exp(-2*elast+x-teta)); 
        
    }
    
    
    public double frandom(int min, int max){
        
        return Math.random()*(max - min) + min;
        
    }
    
    public void propagaTeste(double vetorX[]){
        
        double sum2;
        
        if(vetorX.length != nInp){
            System.out.println("Tamanho da entrada diverge");
            return;
        }
        
        for(int j = 0; j < nInp; j++){
            System.out.println(vetorX[j]);
        }
        
        for(int i = 0; i < nHid; i++){
            sum2 = 0;
            for(int j = 0; j < nOut; j++){
                sum2 += hidW[i][j]*inpA[j];
            }
            hidA[i] = tamh(sum2);
        }
        
        for(int i = 0; i < nOut; i++){
            sum2 = 0;
            for(int j = 0; j < nHid; j++){
                sum2 += outW[i][j] * hidA[j];
            }
            outN[i] = tamh(sum2);
            if(outN[i] > 0){
                System.out.println("Verdadeiro");
            }else{
                System.out.println("Falso");
            }
            
        }
        
       
        
        
    }
    
    public void aprendizado(double in[], double out[]){
        //Método chamado pela classe principal para realizar o aprendizado 
        
        //Conferir se o tamanho de in é igual ao nInput
        if(in.length != nInp){
            return;
        }
        
        if(out.length != nOut){
            return;
        }
        
        inpA = in;
        outA = out;
        
        this.feedForward();
        erroTotal = 0;
        for(int j = 0; j < nOut; j++){ // Percorrendo os neuronios da camada de saida para calcular o delta(erro)
            
            //Nesse calcula delta tambem atualiza os pesos entre a camada oculta e a camada de saida
            erroTotal += this.calculaDelta(j);
            
            
        }
        //Funcao para atualizar os pesos entre a camada oculta e a camada de entrada
        this.atualizaPesos();
        
        System.out.println("Erro total: "+erroTotal);
    }
    
    public void atualizaPesos(){
        
        for(int i = 0; i < nHid ; i++){
            
            double sum = 0;
            for(int n = 0; n < nOut; n++){
                sum += outD[n] * outW[i][n];
            }
            for(int j = 0; j < nInp; j++){ // Atualizando pesos entre a camada de entrada e de saída
                
                hidW[i][j] += eida * sum * inpA[j]; // Backpropagation
                
            }
        }
        
        
        
    }
    
    
    public double calculaDelta(int m){
        //Calcula o erro da posicao
        outD[m] = outA[m] - outN[m]; 
        
        //Atualizando os pesos entre a camada oculta e camada de saida
        
        for(int i = 0; i < nHid; i++){
            outW[m][i] += outD[m] * eida * hidA[i];
        }
        
        return outD[m];
    }
    
    public void feedForward(){
        
        double soma;
        
        for(int i = 0; i < nHid; i++){ // Percorre camada oculta
            soma = 0;
            for(int j = 0; j < nOut; j++){ // Percorre camada de entrada
                soma += hidW[i][j]*inpA[j]; // Multiplica os pesos
            }
            hidA[i] = tamh(soma);
        }
        
        for(int i = 0; i < nOut; i++){
            soma = 0;
            for(int j = 0; j < nHid; j++){
                soma += outW[i][j] * hidA[j];
            }
            this.outN[i] = tamh(soma);
        }
        
    }
    
    public static void main(String[] args) {
        /*Criaremos uma rede neural para o problema XOR (Ou Exclusivo)
            3 entradas
            4 neuronios na camada oculta
            1 neuronio na camada de saída
            taxa de aprendizado 0.02
            limiar da funcao tamh 1
            elasticidade 2
        */
        
        RedeNeural rede = new RedeNeural(3, 4, 1, 0.02, 1, 2);
        
        double FALSO = rede.FALSO;
        double VERDADEIRO = rede.VERDADEIRO;
        
        double entrada[][] = {{VERDADEIRO, FALSO, 1}, {FALSO, VERDADEIRO, 1}, {FALSO, FALSO, 1}, {VERDADEIRO, VERDADEIRO, 1}};
        
        double target[][] = {{FALSO}, {VERDADEIRO}, {VERDADEIRO}, {FALSO}}; 
        
        System.out.println("Saídas antes do aprendizado: ");
        
        for(int i = 0; i < entrada.length; i++){
            rede.propagaTeste(entrada[i]);
        }
        
        // Realizando treinamento
        
        int iteracoes = 1000;
        
        while(iteracoes > 0){
            
            rede.aprendizado(entrada[iteracoes%entrada.length], target[iteracoes%entrada.length]); //Manda linha por linha
            
            iteracoes--;
        }
        
        
    }
    
}
