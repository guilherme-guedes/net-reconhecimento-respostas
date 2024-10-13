using ReconhecimentoDeRespostas;

Console.WriteLine("Hello, World!");

int questoes = 100;
int qtdAlternativas = 5;
int qtdColunasCartao = 5;

int[] gabarito = GabaritoQuestoes(questoes, qtdAlternativas);

var corretor = new Corretor(questoes: questoes, alternativas: qtdAlternativas, gabarito, colunas: qtdColunasCartao);
var cartaoResposta = corretor.Corrigir(@"cartao-vermelho-marcado.png", true); // alterar para caminho da imagem

Console.ReadKey();


static int[] GabaritoQuestoes(int qtdQuestoes, int qtdAlternativas)
{
    int[] gabarito = new int[qtdQuestoes];

    for (int i = 0; i < qtdQuestoes; i++)
        gabarito[i] = Random.Shared.Next(qtdAlternativas - 1);

    return gabarito;
}