using OpenCvSharp;
using System;
using static IdCardAndPictureCheck.Form1;
using Point = OpenCvSharp.Point;

public class IdLiveFaceMatcher : IDisposable
{
    private readonly ArcFaceEmbedder _embedder;

    public IdLiveFaceMatcher(
        string onnxModelPath = @"models\arcface.onnx",
        string haarCascadePath = @"assets\haarcascade_frontalface_default.xml")
    {
        _embedder = new ArcFaceEmbedder(onnxModelPath, haarCascadePath);
    }

    public void Dispose() => _embedder.Dispose();

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

        double bestSim = double.NegativeInfinity;
        int bestIndex = -1;

        for (int i = 0; i < camFaces.Length; i++)
        {
            var emb = camFaces[i].embedding;
            double sim = ArcFaceEmbedder.CosineSimilarity(idEmb, emb);

            if (sim > bestSim)
            {
                bestSim = sim;
                bestIndex = i;
            }
        }

        bool isSame = bestSim >= threshold;
        Rect bestRect = bestIndex >= 0 ? camFaces[bestIndex].rect : new Rect(0, 0, 0, 0);

        // 3. Annotate camera image in memory, return JPEG bytes
        byte[] jpegBytes;
        using (var img = Cv2.ImRead(cameraImagePath))
        {
            if (bestRect.Width > 0 && bestRect.Height > 0)
            {
                // Draw green rectangle
                Cv2.Rectangle(img, bestRect, Scalar.LimeGreen, 2);

                // Draw similarity label
                string label = $"{bestSim:F3}";
                int baseLine;
                var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.7, 1, out baseLine);
                var textOrg = new Point(bestRect.X, bestRect.Y - 5);
                if (textOrg.Y < textSize.Height) textOrg.Y = bestRect.Y + textSize.Height + 5;

                Cv2.PutText(img, label, textOrg,
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
            }

            // Convert annotated Mat → JPEG byte[]
            jpegBytes = img.ImEncode(".jpg");
        }

        return new FaceMatchResult
        {
            IsSamePerson = isSame,
            BestSimilarity = bestSim,
            MatchedFaceRect = bestRect,
            AnnotatedImageBytes = jpegBytes
        };
    }
}
