using OpenCvSharp;
using System;
using static IdCardAndPictureCheck.Form1;
using Point = OpenCvSharp.Point;

public class IdLiveFaceMatcher : IDisposable
{
    private readonly ArcFaceEmbedder _embedder;
    private readonly AntiSpoofingDetector _antiSpoofing;

    public IdLiveFaceMatcher(
        string onnxModelPath = @"models\arcface.onnx",
        string haarCascadePath = @"assets\haarcascade_frontalface_default.xml",
        string antiSpoofPrototxtPath = @"models\deploy_Squeeze.prototxt",
        string antiSpoofCaffemodelPath = @"models\train_add_data_iter_100000.caffemodel")
    {
        _embedder = new ArcFaceEmbedder(onnxModelPath, haarCascadePath);
        _antiSpoofing = new AntiSpoofingDetector(antiSpoofPrototxtPath, antiSpoofCaffemodelPath);
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

        double bestSim = double.NegativeInfinity;
        int bestIndex = -1;
        double antiSpoofConfidence = 0;
        bool isLiveFace = false;

        using var cameraBgr = Cv2.ImRead(cameraImagePath);
        if (cameraBgr.Empty())
            throw new InvalidOperationException("Unable to read camera image: " + cameraImagePath);

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

        Rect bestRect = bestIndex >= 0 ? camFaces[bestIndex].rect : new Rect(0, 0, 0, 0);
        if (bestIndex >= 0)
        {
            var antiSpoof = _antiSpoofing.Evaluate(cameraBgr, bestRect);
            isLiveFace = antiSpoof.isLive;
            antiSpoofConfidence = antiSpoof.confidence;
        }

        bool isSame = bestSim >= threshold && isLiveFace;

        // 3. Annotate camera image in memory, return JPEG bytes
        byte[] jpegBytes;
        using (var img = cameraBgr.Clone())
        {
            if (bestRect.Width > 0 && bestRect.Height > 0)
            {
                var boxColor = isLiveFace ? Scalar.LimeGreen : Scalar.Red;
                Cv2.Rectangle(img, bestRect, boxColor, 2);

                string label = isLiveFace ? $"Live {antiSpoofConfidence:F3}" : $"Spoof {antiSpoofConfidence:F3}";
                int baseLine;
                var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.7, 1, out baseLine);
                var textOrg = new Point(bestRect.X, bestRect.Y - 5);
                if (textOrg.Y < textSize.Height) textOrg.Y = bestRect.Y + textSize.Height + 5;

                Cv2.PutText(img, label, textOrg, HersheyFonts.HersheySimplex, 0.7, boxColor, 2);

                string simLabel = $"{bestSim:F3}";
                var simOrg = new Point(bestRect.X, textOrg.Y + textSize.Height + 5);
                Cv2.PutText(img, simLabel, simOrg, HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
            }

            // Convert annotated Mat → JPEG byte[]
            jpegBytes = img.ImEncode(".jpg");
        }

        return new FaceMatchResult
        {
            IsSamePerson = isSame,
            BestSimilarity = bestSim,
            MatchedFaceRect = bestRect,
            IsLiveFace = isLiveFace,
            AntiSpoofConfidence = antiSpoofConfidence,
            AnnotatedImageBytes = jpegBytes
        };
    }
}
