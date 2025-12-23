using IDCardFaceMatchHelper64.DTOs;
using IDCardFaceMatchHelper64.Helper;
using OpenCvSharp;
using System;
using Point = OpenCvSharp.Point;

namespace IDCardFaceMatchHelper64
{
    public class IdLiveFaceMatcher : IDisposable
    {
        private readonly ArcFaceEmbedder _embedder;
        private readonly AntiSpoofingDetector _antiSpoofing;

        public IdLiveFaceMatcher()
        {
            //string onnxPath = Path.Combine(AppContext.BaseDirectory, "models", "arcface.onnx");
            //string cascadePath = Path.Combine(AppContext.BaseDirectory, "assets", "haarcascade_frontalface_default.xml");

            string onnxPath = EmbeddedResourceHelper.ExtractToTemp("arcface.onnx");
            string cascadePath = EmbeddedResourceHelper.ExtractToTemp("haarcascade_frontalface_default.xml");


            string antiSpoofPrototxt = EmbeddedResourceHelper.ExtractToTemp("deploy_Squeeze.prototxt");
            string antiSpoofModel = EmbeddedResourceHelper.ExtractToTemp("train_add_data_iter_100000.caffemodel");

            _embedder = new ArcFaceEmbedder(onnxModelPath: onnxPath, haarCascadePath: cascadePath);
            _antiSpoofing = new AntiSpoofingDetector(antiSpoofPrototxt, antiSpoofModel);
        }

        public void Dispose()
        {
            _embedder.Dispose();
            _antiSpoofing.Dispose();
        }

        public FaceMatchResult MatchIdToCamera(
            string idCardPath,
            string cameraImagePath,
            double threshold = 0.40)
        {
            // 1. Embedding for ID card face
            var idEmb = _embedder.GetEmbeddingFromFile(idCardPath);

            // 2. Embeddings for all faces in camera image
            var camFaces = _embedder.GetAllFaceEmbeddings(cameraImagePath);
            if (camFaces.Length == 0)
                throw new Exception("No faces detected in camera image.");

            double bestLiveSim = double.NegativeInfinity;
            int bestLiveIndex = -1;
            double bestSpoofConfidence = double.NegativeInfinity;
            int bestSpoofIndex = -1;
            double bestSpoofSim = double.NegativeInfinity;
            double bestLiveAntiConf = 0;

            using var cameraBgr = Cv2.ImRead(cameraImagePath);
            if (cameraBgr.Empty())
                throw new InvalidOperationException("Unable to read camera image: " + cameraImagePath);

            for (int i = 0; i < camFaces.Length; i++)
            {
                var emb = camFaces[i].embedding;
                double sim = ArcFaceEmbedder.CosineSimilarity(idEmb, emb);
                var (isLive, antiConf) = _antiSpoofing.Evaluate(cameraBgr, camFaces[i].rect);

                if (isLive)
                {
                    if (sim > bestLiveSim)
                    {
                        bestLiveSim = sim;
                        bestLiveIndex = i;
                        bestLiveAntiConf = antiConf;
                    }
                }
                else if (antiConf > bestSpoofConfidence)
                {
                    bestSpoofConfidence = antiConf;
                    bestSpoofIndex = i;
                    bestSpoofSim = sim;
                }
            }

            bool hasLiveFace = bestLiveIndex >= 0;
            bool isSame = hasLiveFace && bestLiveSim >= threshold;
            Rect bestRect = hasLiveFace ? camFaces[bestLiveIndex].rect
                                        : (bestSpoofIndex >= 0 ? camFaces[bestSpoofIndex].rect : new Rect(0, 0, 0, 0));

            // 3. Annotate camera image in memory, return JPEG bytes
            byte[] jpegBytes;
            using (var img = cameraBgr.Clone())
            {
                if (bestRect.Width > 0 && bestRect.Height > 0)
                {
                    var boxColor = hasLiveFace ? Scalar.LimeGreen : Scalar.Red;
                    Cv2.Rectangle(img, bestRect, boxColor, 2);

                    string label = hasLiveFace ? $"Live {bestLiveAntiConf:F3}" : $"Spoof {bestSpoofConfidence:F3}";
                    int baseLine;
                    var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.7, 1, out baseLine);
                    var textOrg = new Point(bestRect.X, bestRect.Y - 5);
                    if (textOrg.Y < textSize.Height) textOrg.Y = bestRect.Y + textSize.Height + 5;

                    Cv2.PutText(img, label, textOrg, HersheyFonts.HersheySimplex, 0.7, boxColor, 2);

                    if (hasLiveFace)
                    {
                        string simLabel = $"{bestLiveSim:F3}";
                        var simOrg = new Point(bestRect.X, textOrg.Y + textSize.Height + 5);
                        Cv2.PutText(img, simLabel, simOrg, HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
                    }
                }

                // Convert annotated Mat → JPEG byte[]
                jpegBytes = img.ImEncode(".jpg");
            }

            return new FaceMatchResult
            {
                IsSamePerson = isSame,
                BestSimilarity = hasLiveFace ? bestLiveSim : bestSpoofSim,
                MatchedFaceRect = bestRect,
                IsLiveFace = hasLiveFace,
                AntiSpoofConfidence = hasLiveFace ? bestLiveAntiConf : bestSpoofConfidence,
                AnnotatedImageBytes = jpegBytes
            };
        }
    }
}
