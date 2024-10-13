using OpenCvSharp;

namespace ReconhecimentoDeRespostas
{
    public class Corretor
    {
        private const int PERCENTUAL_MIN_PREENCHIMENTO_ALTERNATIVA = 55;
        private readonly int _colunas;
        private readonly int _quantidadeQuestoes;
        private readonly int _quantidadeAlternativasPorQuestao;
        private readonly int[] _gabarito;
        private readonly bool debugarTelaCinza = false;
        private readonly bool debugarTelaPretoBranco = false;
        private readonly bool debugarTelaInvertida = false;
        private readonly bool debugarTelaOriginal = false;

        public Corretor(int questoes, int alternativas, int[] gabarito, int colunas)
        {
            _colunas = colunas;
            _quantidadeQuestoes = questoes;
            _quantidadeAlternativasPorQuestao = alternativas;
            this._gabarito = gabarito;
        }

        public Dictionary<int, ResultadoCorrecaoDTO> Corrigir(string imagemCartaoResposta, bool debug = false)
        {
            Mat imagemOriginal = new Mat(imagemCartaoResposta);
            Mat imagemCinza = new Mat();
            Mat imgInvertida = new Mat();
            Mat imagemPretoBranco = new Mat();
            try
            {
                // Tratamentos iniciais
                TransformarParaImagemCinza(fonte: imagemOriginal, destino: imagemCinza);
                TransformarParaBinario(fonte: imagemCinza, destino: imagemPretoBranco);

                // Aplicação de filtros
                //Cv2.Blur(src: imagemPretoBranco, dst: imagemPretoBranco, new Size(3, 3));
                Cv2.GaussianBlur(src: imagemPretoBranco, dst: imagemPretoBranco, ksize: new Size(3, 3), sigmaX: 0.7, sigmaY: 0.7);

                // Obtenção das formas das alternativas e matrização
                var circulosAlternativas = ObterCirculosAlternativas(imagemPretoBranco, paramMin: 12, paramMax: 13, distanciaMinima: 22);
                CircleSegment[][] questoesComAlternativas = MatrizarEOrdenarAlternativas(circulosAlternativas);

                // Realce das formas
                ContornarCirculosAlternativas(imagemCinza, circulosAlternativas, Scalar.Black);

                TransformarParaBinario(fonte: imagemCinza, destino: imagemPretoBranco);
                TransformarParaInverso(fonte: imagemPretoBranco, destino: imgInvertida);

                Dictionary<int, ResultadoCorrecaoDTO> resultadoCartaoResposta = VerificarAlternativas(imgBinariaInvertida: imgInvertida, circulosQuestoes: questoesComAlternativas);

                if (debug)
                {
                    if (debugarTelaCinza)
                        MostrarTela("Cinza", imagemCinza);
                    if (debugarTelaPretoBranco)
                        MostrarTela("Preto&Branco", imagemPretoBranco);

                    //MostrarImagensAlternativasDebug(imgInvertida, questoesComAlternativas);
                    ContornarCirculosAlternativas(imagemOriginal, circulosAlternativas, Scalar.Green);
                    ExibirImagensFinaisDebug(imagemOriginal, imgInvertida, questoesComAlternativas, resultadoCartaoResposta);
                }

                return resultadoCartaoResposta;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            finally
            {
                Cv2.DestroyAllWindows();
                imgInvertida?.Dispose();
                imagemCinza?.Dispose();
                imagemOriginal?.Dispose();
                imagemPretoBranco?.Dispose();
            }

            return null;
        }

        private void ExibirImagensFinaisDebug(Mat imagemOriginal, Mat imgInvertida, CircleSegment[][] questoesComAlternativas, Dictionary<int, ResultadoCorrecaoDTO> resultado)
        {
            Mat imgCartao = null;
            Mat imgGabarito = null;
            if (questoesComAlternativas is not null)
            {
                imgCartao = CriarImagemVerificacaoCartao(imgInvertida, questoesComAlternativas, resultado);
                MostrarTela("Cartão confirmação", imgCartao);

                imgGabarito = CriarImagemGabarito(imgInvertida, questoesComAlternativas, _gabarito);
                MostrarTela("Gabarito", imgGabarito);
            }

            if (debugarTelaOriginal)
                MostrarTela("Original", imagemOriginal);

            if (debugarTelaInvertida)
                MostrarTela("invertida", imgInvertida);

            Cv2.WaitKey();

            imgCartao?.Dispose();
            imgGabarito?.Dispose();
        }

        private Dictionary<int, ResultadoCorrecaoDTO> VerificarAlternativas(Mat imgBinariaInvertida, CircleSegment[][] circulosQuestoes)
        {
            Mat imgAlternativa = null;
            Dictionary<int, ResultadoCorrecaoDTO> resultado = new Dictionary<int, ResultadoCorrecaoDTO>();
            try
            {
                for (int iQuestao = 0; iQuestao < _quantidadeQuestoes; iQuestao++)
                {
                    resultado[iQuestao] = new ResultadoCorrecaoDTO(iQuestao, _gabarito[iQuestao]);

                    for (int iAlternativa = 0; iAlternativa < _quantidadeAlternativasPorQuestao; iAlternativa++)
                    {
                        imgAlternativa = new Mat(imgBinariaInvertida, ExtrairRetanguloAlternativa(circulosQuestoes[iQuestao][iAlternativa]));
                        if (AlternativaMarcada(imgAlternativa))
                        {
                            if (resultado[iQuestao].TemAlternativaMarcada())
                            {
                                resultado[iQuestao].AlternativaMarcada = null;
                                break;
                            }

                            resultado[iQuestao].AlternativaMarcada = iAlternativa;
                            continue;
                        }
                    }
                }
            }
            finally
            {
                imgAlternativa?.Dispose();
            }
            return resultado;
        }

        private bool AlternativaMarcada(Mat roiAlternativa)
        {
            int qtdAtivos = Cv2.CountNonZero(roiAlternativa);
            decimal proporcao = decimal.Divide(qtdAtivos, roiAlternativa.Height * roiAlternativa.Width);
            return Math.Round(proporcao * 100, 2) > PERCENTUAL_MIN_PREENCHIMENTO_ALTERNATIVA;
        }

        private CircleSegment[] ObterCirculosAlternativas(Mat imagemCinza,
            int raioMinimo = 9,
            int raioMaximo = 12,
            int paramMin = 10,
            int paramMax = 20,
            int distanciaMinima = 16)
        {
            var circulosAlternativas = Cv2.HoughCircles(image: imagemCinza,
                            method: HoughModes.Gradient,
                            dp: 1.0,
                            minDist: distanciaMinima,
                            param1: paramMin,
                            param2: paramMax,
                            minRadius: raioMinimo,
                            maxRadius: raioMaximo);

            return circulosAlternativas.OrderBy(x => x.Center.Y).ToArray();
        }

        private CircleSegment[][] MatrizarEOrdenarAlternativas(CircleSegment[] circulosAlternativas)
        {
            var qtdTotalEncontrada = circulosAlternativas.Count();
            int qtdTotal = _quantidadeQuestoes * _quantidadeAlternativasPorQuestao;
            int qtdPrimeiraLinha = _colunas * _quantidadeAlternativasPorQuestao;
            var qtdLinhas = (int)Math.Round((decimal)_quantidadeQuestoes / _colunas, 0, MidpointRounding.ToPositiveInfinity); // jogar pra cima

            if (qtdTotalEncontrada != qtdTotal)
                throw new Exception($"A quantidade de círculos/alternativas encontradas ({qtdTotalEncontrada}) não é igual a quantidade total informada ({_quantidadeQuestoes} questões * {_quantidadeAlternativasPorQuestao} alternativas).");

            #region Segmentação e organização de circulos/alternativas por linha (Y)

            List<CircleSegment[]> alternativasPorLinha = new List<CircleSegment[]>();
            CircleSegment[] linhaIteradora = circulosAlternativas.Take(qtdPrimeiraLinha).OrderBy(c => c.Center.X).ToArray();
            alternativasPorLinha.Add(linhaIteradora);
            circulosAlternativas = circulosAlternativas.Skip(qtdPrimeiraLinha).ToArray();

            var iLinha = 1;
            do
            {
                var maxYAtual = circulosAlternativas[0].Center.Y + circulosAlternativas[0].Radius;

                linhaIteradora = circulosAlternativas.Where(c => c.Center.Y <= maxYAtual).OrderBy(c => c.Center.X).ToArray();
                alternativasPorLinha.Add(linhaIteradora);

                circulosAlternativas = circulosAlternativas.Skip(linhaIteradora.Length).ToArray();
                iLinha++;
            } while (iLinha < qtdLinhas);

            #endregion Segmentação e organização de circulos/alternativas por linha (Y)

            #region Ordenação de circulos/alternativas para segmentar por ordem das questões

            CircleSegment[][] questoesOrdenadas = new CircleSegment[_quantidadeQuestoes][];
            iLinha = 0;
            for (int iQuestao = 0, iColuna = 0; iQuestao < _quantidadeQuestoes; iQuestao++, iLinha++)
            {
                if (iLinha == qtdLinhas)
                {
                    iColuna++;
                    iLinha = 0;
                }

                CircleSegment[] alternativasQuestao = alternativasPorLinha[iLinha].Skip(iColuna * _quantidadeAlternativasPorQuestao).Take(_quantidadeAlternativasPorQuestao).ToArray();
                questoesOrdenadas[iQuestao] = alternativasQuestao;
            }

            #endregion Ordenação de circulos/alternativas para segmentar por ordem das questões

            return questoesOrdenadas;
        }

        #region Transformações

        private void TransformarParaImagemCinza(Mat fonte, Mat destino)
        {
            Cv2.CvtColor(src: fonte, dst: destino, ColorConversionCodes.BGR2GRAY);
        }

        private void TransformarParaBinario(Mat fonte, Mat destino)
        {
            Cv2.Threshold(src: fonte, dst: destino, 1, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
        }

        private void TransformarParaInverso(Mat fonte, Mat destino)
        {
            Cv2.BitwiseNot(src: fonte, dst: destino);
        }

        #endregion Transformações

        #region Exibição

        internal void MostrarImagensAlternativasDebug(Mat imgInvertida, CircleSegment[][] questoesComAlternativas)
        {
            // 0 x N
            Rect roi00 = ExtrairRetanguloAlternativa(questoesComAlternativas[0][0]);
            Rect roi01 = ExtrairRetanguloAlternativa(questoesComAlternativas[0][1]);
            Rect roi02 = ExtrairRetanguloAlternativa(questoesComAlternativas[0][2]);
            Rect roi03 = ExtrairRetanguloAlternativa(questoesComAlternativas[0][3]);
            Rect roi04 = ExtrairRetanguloAlternativa(questoesComAlternativas[0][4]);

            // 1 x N
            Rect roi10 = ExtrairRetanguloAlternativa(questoesComAlternativas[1][0]);
            Rect roi11 = ExtrairRetanguloAlternativa(questoesComAlternativas[1][1]);
            Rect roi12 = ExtrairRetanguloAlternativa(questoesComAlternativas[1][2]);
            Rect roi13 = ExtrairRetanguloAlternativa(questoesComAlternativas[1][3]);
            Rect roi14 = ExtrairRetanguloAlternativa(questoesComAlternativas[1][4]);

            // 2 x N
            Rect roi20 = ExtrairRetanguloAlternativa(questoesComAlternativas[2][0]);
            Rect roi21 = ExtrairRetanguloAlternativa(questoesComAlternativas[2][1]);
            Rect roi22 = ExtrairRetanguloAlternativa(questoesComAlternativas[2][2]);
            Rect roi23 = ExtrairRetanguloAlternativa(questoesComAlternativas[2][3]);
            Rect roi24 = ExtrairRetanguloAlternativa(questoesComAlternativas[2][4]);

            // 3 x N
            Rect roi30 = ExtrairRetanguloAlternativa(questoesComAlternativas[3][0]);
            Rect roi31 = ExtrairRetanguloAlternativa(questoesComAlternativas[3][1]);
            Rect roi32 = ExtrairRetanguloAlternativa(questoesComAlternativas[3][2]);
            Rect roi33 = ExtrairRetanguloAlternativa(questoesComAlternativas[3][3]);
            Rect roi34 = ExtrairRetanguloAlternativa(questoesComAlternativas[3][4]);

            MostrarTela("0x0", new Mat(imgInvertida, roi00));
            MostrarTela("0x1", new Mat(imgInvertida, roi01));
            MostrarTela("0x2", new Mat(imgInvertida, roi02));
            MostrarTela("0x3", new Mat(imgInvertida, roi03));
            MostrarTela("0x4", new Mat(imgInvertida, roi04));

            MostrarTela("1x0", new Mat(imgInvertida, roi10));
            MostrarTela("1x1", new Mat(imgInvertida, roi11));
            MostrarTela("1x2", new Mat(imgInvertida, roi12));
            MostrarTela("1x3", new Mat(imgInvertida, roi13));
            MostrarTela("1x4", new Mat(imgInvertida, roi14));

            MostrarTela("2x0", new Mat(imgInvertida, roi20));
            MostrarTela("2x1", new Mat(imgInvertida, roi21));
            MostrarTela("2x2", new Mat(imgInvertida, roi22));
            MostrarTela("2x3", new Mat(imgInvertida, roi23));
            MostrarTela("2x4", new Mat(imgInvertida, roi24));

            MostrarTela("3x0", new Mat(imgInvertida, roi30));
            MostrarTela("3x1", new Mat(imgInvertida, roi31));
            MostrarTela("3x2", new Mat(imgInvertida, roi32));
            MostrarTela("3x3", new Mat(imgInvertida, roi33));
            MostrarTela("3x4", new Mat(imgInvertida, roi34));
        }

        private static void DesenharAlternativaNaoMarcada(Mat imgCartao, CircleSegment alternativa)
        {
            Cv2.Circle(img: imgCartao,
                                            center: alternativa.Center.ToPoint(),
                                            radius: (int)alternativa.Radius,
                                            color: Scalar.White,
                                            thickness: 1);
        }

        private static void DesenharAlternativaMarcada(Mat imgCartao, CircleSegment alternativa)
        {
            Cv2.Circle(img: imgCartao,
                                            center: alternativa.Center.ToPoint(),
                                            radius: (int)alternativa.Radius,
                                            color: Scalar.White,
                                            thickness: -1);
        }

        private void ContornarCirculosAlternativas(Mat imagemFonte, CircleSegment[] circulos, Scalar cor, int espessura = 1)
        {
            int i = 0;
            foreach (var circulo in circulos)
            {
                // teste
                //if (i >= 245) i = 0;
                //var corAux = new Scalar(i, i, i);
                //i += 20;
                // fim-teste

                Cv2.Circle(img: imagemFonte,
                    centerX: (int)circulo.Center.X,
                    centerY: (int)circulo.Center.Y,
                    radius: (int)circulo.Radius,
                    color: cor, // corAux
                    thickness: espessura); // espessura
            }
        }

        private Rect ExtrairRetanguloAlternativa(CircleSegment alternativa)
        {
            return new Rect(
                                X: (int)(alternativa.Center.X - alternativa.Radius),
                                Y: (int)(alternativa.Center.Y - alternativa.Radius),
                                Width: (int)alternativa.Radius * 2 + 1, (int)alternativa.Radius * 2 + 1);
        }

        private Mat CriarImagemGabarito(Mat imgReferencia, CircleSegment[][] questoesComAlternativas, int[] gabarito)
        {
            if (questoesComAlternativas is not null && questoesComAlternativas.Length > 0)
            {
                Mat imgCartaoGabarito = new Mat(imgReferencia.Size(), imgReferencia.Type(), Scalar.Black);
                for (int iQuestao = 0; iQuestao < questoesComAlternativas.Length; iQuestao++)
                {
                    for (int iAlternativa = 0; iAlternativa < _quantidadeAlternativasPorQuestao; iAlternativa++)
                    {
                        var alternativa = questoesComAlternativas[iQuestao][iAlternativa];
                        if (gabarito[iQuestao] == iAlternativa)
                            DesenharAlternativaMarcada(imgCartaoGabarito, alternativa);
                        else
                            DesenharAlternativaNaoMarcada(imgCartaoGabarito, alternativa);
                    }
                }

                return imgCartaoGabarito;
            }
            return null;
        }

        private Mat CriarImagemVerificacaoCartao(Mat imgReferencia, CircleSegment[][] questoesComAlternativas, Dictionary<int, ResultadoCorrecaoDTO> resultado)
        {
            if (questoesComAlternativas is not null && questoesComAlternativas.Length > 0)
            {
                var tamanho = imgReferencia.Size();
                Mat imgCartao = new Mat(tamanho.Height + 30, tamanho.Width, imgReferencia.Type(), Scalar.Black);
                int acertos = 0;
                for (int iQuestao = 0; iQuestao < questoesComAlternativas.Length; iQuestao++)
                {
                    for (int iAlternativa = 0; iAlternativa < _quantidadeAlternativasPorQuestao; iAlternativa++)
                    {
                        var alternativa = questoesComAlternativas[iQuestao][iAlternativa];
                        if (resultado[iQuestao].AlternativaMarcada.HasValue &&
                            resultado[iQuestao].AlternativaMarcada!.Value == iAlternativa)
                            DesenharAlternativaMarcada(imgCartao, alternativa);
                        else
                            DesenharAlternativaNaoMarcada(imgCartao, alternativa);
                    }
                    if (resultado[iQuestao].Acertou())
                        acertos++;
                }

                Cv2.PutText(imgCartao, $"Acertos: {acertos}", new Point(X: 5, Y: tamanho.Height + 20), HersheyFonts.HersheyPlain, 1, Scalar.White, 1);

                return imgCartao;
            }
            return null;
        }

        private void MostrarTela(string tituloTela, Mat imagemExibida)
        {
            Cv2.ImShow(tituloTela, imagemExibida);
        }

        #endregion Exibição

        #region Teste

        private RetornoContornosDTO ObterContornosAlternativas(Mat imagemFonte)
        {
            Point[][] contornos = null;
            HierarchyIndex[] indices = null;

            Cv2.FindContours(image: imagemFonte,
                contours: out contornos,
                hierarchy: out indices,
                mode: RetrievalModes.List,
                method: ContourApproximationModes.ApproxSimple);

            if (contornos is null) return null;

            return new RetornoContornosDTO(contornos.Where(c => c.Length > 10).ToArray(), indices);
        }

        #endregion Teste
    }

    #region Classes Auxiliares

    internal record RetornoContornosDTO(Point[][] Contornos, HierarchyIndex[] Indices);

    public class ResultadoCorrecaoDTO
    {
        public int IndiceQuestao { get; private set; }
        public int AlternativaCorreta { get; private set; }
        public int? AlternativaMarcada { get; set; }

        public ResultadoCorrecaoDTO(int indiceQuestao, int alternativaCorreta, int? alternativaMarcada = null)
        {
            this.IndiceQuestao = indiceQuestao;
            this.AlternativaCorreta = alternativaCorreta;
            this.AlternativaMarcada = alternativaMarcada;
        }

        public bool TemAlternativaMarcada() => this.AlternativaMarcada.HasValue;

        public bool Acertou() => this.TemAlternativaMarcada() && this.AlternativaMarcada!.Value == this.AlternativaCorreta;
    }

    #endregion Classes Auxiliares
}