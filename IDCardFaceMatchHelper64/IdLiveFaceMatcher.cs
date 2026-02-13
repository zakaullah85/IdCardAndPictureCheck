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

        private readonly float _faceConfThreshold;

        public IdLiveFaceMatcher(double faceConfThreshold = 0.5, double liveThreshold = 0.95)
        {
            string arcfacePath = EmbeddedResourceHelper.ExtractToTemp("arcface.onnx");
            string scrfdPath = EmbeddedResourceHelper.ExtractToTemp("scrfd.onnx");

            string antiSpoofPrototxt = EmbeddedResourceHelper.ExtractToTemp("deploy_Squeeze.prototxt");
            string antiSpoofModel = EmbeddedResourceHelper.ExtractToTemp("train_add_data_iter_100000.caffemodel");

            _faceConfThreshold = (float)faceConfThreshold;

            _embedder = new ArcFaceEmbedder(
                arcfaceOnnxPath: arcfacePath,
                scrfdOnnxPath: scrfdPath,
                faceConfThreshold: _faceConfThreshold,
                faceNmsIou: 0.35f);

            _antiSpoofing = new AntiSpoofingDetector(antiSpoofPrototxt, antiSpoofModel, liveThreshold);
        }

        public void Dispose()
        {
            _embedder.Dispose();
            _antiSpoofing.Dispose();
        }

        public FaceMatchResult MatchIdToCamera(string idCardPath, string cameraImagePath, double threshold = 0.40)
        {
            var idEmb = _embedder.GetEmbeddingFromFile(idCardPath);

            var camFaces = _embedder.GetAllFaceEmbeddings(cameraImagePath);
            if (camFaces.Length == 0)
                throw new Exception("No faces detected in camera image.");

            double bestSim = double.NegativeInfinity;
            int bestIndex = -1;

            using var cameraBgr = Cv2.ImRead(cameraImagePath);
            if (cameraBgr.Empty())
                throw new InvalidOperationException("Unable to read camera image: " + cameraImagePath);

            for (int i = 0; i < camFaces.Length; i++)
            {
                double sim = ArcFaceEmbedder.CosineSimilarity(idEmb, camFaces[i].embedding);
                if (sim > bestSim)
                {
                    bestSim = sim;
                    bestIndex = i;
                }
            }

            Rect bestRect = bestIndex >= 0 ? camFaces[bestIndex].rect : new Rect(0, 0, 0, 0);

            bool isLiveFace = false;
            double antiSpoofConfidence = 0;

            if (bestIndex >= 0)
            {
                var antiSpoof = _antiSpoofing.Evaluate(cameraBgr, bestRect);
                isLiveFace = antiSpoof.isLive;
                antiSpoofConfidence = antiSpoof.confidence;
            }

            bool isSame = bestSim >= threshold && isLiveFace;

            byte[] jpegBytes;
            using (var img = cameraBgr.Clone())
            {
                if (bestRect.Width > 0 && bestRect.Height > 0)
                {
                    Scalar boxColor = Scalar.Red;
                    if (isLiveFace && isSame) boxColor = Scalar.Lime;
                    else if (isLiveFace && !isSame) boxColor = Scalar.Orange;

                    var textColor = Scalar.Black;
                    var bgColor = boxColor;

                    Cv2.Rectangle(img, bestRect, boxColor, 2);

                    double fontScale = 0.7;
                    int thickness = 1;
                    int padding = 4;
                    int gap = 6;

                    string labelLeft = $"face con.{_faceConfThreshold:F2}";
                    string topLabel = isLiveFace ? $"Live {antiSpoofConfidence:F3}" : $"Spoof {antiSpoofConfidence:F3}";
                    string bottomLabel = $"Sim {bestSim:F3}";

                    // LEFT vertical label
                    int leftBaseLine;
                    var leftSize = Cv2.GetTextSize(labelLeft, HersheyFonts.HersheySimplex, fontScale, thickness, out leftBaseLine);

                    int textW = leftSize.Width + padding * 2;
                    int textH = leftSize.Height + leftBaseLine + padding * 2;

                    using var labelMat = new Mat(textH, textW, MatType.CV_8UC3, Scalar.All(0));
                    labelMat.SetTo(bgColor);

                    Cv2.PutText(labelMat, labelLeft, new Point(padding, textH - padding - leftBaseLine),
                        HersheyFonts.HersheySimplex, fontScale, textColor, thickness);

                    using var rotated = new Mat();
                    Cv2.Rotate(labelMat, rotated, RotateFlags.Rotate90Counterclockwise);

                    int leftX = bestRect.X - rotated.Width - gap;
                    int leftY = bestRect.Y + (bestRect.Height - rotated.Height) / 2;

                    leftX = Math.Max(0, leftX);
                    leftY = Math.Max(0, leftY);

                    var roiRect = new Rect(leftX, leftY,
                        Math.Min(rotated.Width, img.Width - leftX),
                        Math.Min(rotated.Height, img.Height - leftY));

                    using (var roi = new Mat(img, roiRect))
                    using (var src = new Mat(rotated, new Rect(0, 0, roiRect.Width, roiRect.Height)))
                        src.CopyTo(roi);

                    // TOP label
                    int topBaseLine;
                    var topSize = Cv2.GetTextSize(topLabel, HersheyFonts.HersheySimplex, fontScale, thickness, out topBaseLine);
                    int topX = bestRect.X + (bestRect.Width - topSize.Width) / 2;
                    int topTextY = bestRect.Y - gap;

                    int topBgY = topTextY - topSize.Height - padding;
                    if (topBgY < 0)
                    {
                        topTextY = bestRect.Y + topSize.Height + padding + gap;
                        topBgY = topTextY - topSize.Height - padding;
                    }

                    var topBg = new Rect(topX - padding, topBgY,
                        topSize.Width + padding * 2,
                        topSize.Height + topBaseLine + padding * 2);

                    Cv2.Rectangle(img, topBg, bgColor, -1);
                    Cv2.PutText(img, topLabel, new Point(topX, topTextY),
                        HersheyFonts.HersheySimplex, fontScale, textColor, thickness);

                    // BOTTOM label
                    int botBaseLine;
                    var botSize = Cv2.GetTextSize(bottomLabel, HersheyFonts.HersheySimplex, fontScale, thickness, out botBaseLine);

                    int botX = bestRect.X + (bestRect.Width - botSize.Width) / 2;
                    int botTextY = bestRect.Y + bestRect.Height + botSize.Height + gap;

                    var botBg = new Rect(botX - padding,
                        botTextY - botSize.Height - padding,
                        botSize.Width + padding * 2,
                        botSize.Height + botBaseLine + padding * 2);

                    if (botBg.Bottom > img.Rows)
                    {
                        botTextY = bestRect.Y + bestRect.Height - gap;
                        botBg = new Rect(botX - padding,
                            botTextY - botSize.Height - padding,
                            botSize.Width + padding * 2,
                            botSize.Height + botBaseLine + padding * 2);
                    }

                    Cv2.Rectangle(img, botBg, bgColor, -1);
                    Cv2.PutText(img, bottomLabel, new Point(botX, botTextY),
                        HersheyFonts.HersheySimplex, fontScale, textColor, thickness);
                }

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
}