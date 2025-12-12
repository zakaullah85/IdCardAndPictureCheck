using IDCardFaceMatchHelper64.DTOs;
using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace IDCardFaceMatchHelper64
{
    public class IdLiveFaceMatcher : IDisposable
    {
        private readonly ArcFaceEmbedder _embedder;

        public IdLiveFaceMatcher()
        {
            _embedder = new ArcFaceEmbedder(
                onnxModelPath: System.IO.Path.Combine(AppContext.BaseDirectory, "models", "arcface.onnx"),
                haarCascadePath: System.IO.Path.Combine(AppContext.BaseDirectory, "assets", "haarcascade_frontalface_default.xml")
            );
        }

        public void Dispose() => _embedder.Dispose();

        public FaceMatchResult MatchIdToCamera(
            string idCardPath,
            string cameraImagePath,
            double threshold = 0.40)
        {
            // 1) Extract ID face crop (BGR)
            var (idRect, idFace) = _embedder.ExtractLargestFaceBgr(idCardPath, pad: 0.12);
            try
            {
                // 2) Build ID variants
                var variants = IdFacePreprocessor.BuildIdFaceVariants(idFace);

                // 3) Embed all ID variants
                var idEmbeddings = new List<float[]>(variants.Length);
                foreach (var v in variants)
                {
                    try
                    {
                        var emb = _embedder.GetEmbeddingFromFaceMat(v);
                        idEmbeddings.Add(emb);
                    }
                    finally
                    {
                        v.Dispose();
                    }
                }

                // 4) Get camera faces (rect + embedding)
                var camFaces = _embedder.GetAllFaceEmbeddings(cameraImagePath);
                if (camFaces == null || camFaces.Length == 0)
                    throw new Exception("No faces detected in camera image.");

                // 5) Compare best-of-N
                double bestSim = double.NegativeInfinity;
                int bestIdx = -1;

                for (int i = 0; i < camFaces.Length; i++)
                {
                    var camEmb = camFaces[i].embedding;

                    foreach (var idEmb in idEmbeddings)
                    {
                        double sim = ArcFaceEmbedder.CosineSimilarity(idEmb, camEmb);
                        if (sim > bestSim)
                        {
                            bestSim = sim;
                            bestIdx = i;
                        }
                    }
                }

                bool isSame = bestSim >= threshold;
                Rect bestRect = bestIdx >= 0 ? camFaces[bestIdx].rect : new Rect(0, 0, 0, 0);

                // 6) Annotate output image
                byte[] annotatedJpeg;
                using (var img = Cv2.ImRead(cameraImagePath))
                {
                    if (!img.Empty() && bestRect.Width > 0 && bestRect.Height > 0)
                    {
                        Cv2.Rectangle(img, bestRect, Scalar.LimeGreen, 2);
                        var label = $"{bestSim:F3}";
                        Cv2.PutText(img, label,
                            new Point(bestRect.X, Math.Max(25, bestRect.Y - 8)),
                            HersheyFonts.HersheySimplex, 0.8, Scalar.Yellow, 2);
                    }

                    annotatedJpeg = img.ImEncode(".jpg");
                }

                return new FaceMatchResult
                {
                    IsSamePerson = isSame,
                    BestSimilarity = bestSim,
                    MatchedFaceRect = bestRect,
                    AnnotatedImageBytes = annotatedJpeg
                };
            }
            finally
            {
                idFace.Dispose();
            }
        }
    }
}
