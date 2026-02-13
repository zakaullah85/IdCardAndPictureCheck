using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace IDCardFaceMatchHelper64
{
    public sealed class AntiSpoofingDetector : IDisposable
    {
        private readonly Net _net;
        private readonly Size _inputSize = new Size(227, 227);
        private readonly Scalar _mean = new Scalar(90.0, 198.0, 121.0);

     
        private readonly double _strongTrueThreshold;

        public AntiSpoofingDetector(string prototxtPath, string caffemodelPath, double strongTrueThreshold = 0.95)
        {
            if (!File.Exists(prototxtPath))
                throw new FileNotFoundException("Anti-spoofing prototxt not found.", prototxtPath);
            if (!File.Exists(caffemodelPath))
                throw new FileNotFoundException("Anti-spoofing caffemodel not found.", caffemodelPath);

            _net = CvDnn.ReadNetFromCaffe(prototxtPath, caffemodelPath);
            _strongTrueThreshold = strongTrueThreshold;
        }

        public void Dispose() => _net?.Dispose();

        // evaluate expanded and original, then apply the same rule tree.
        public (bool isLive, double confidence) Evaluate(Mat bgrImage, Rect faceRect)
        {
            if (bgrImage.Empty() || faceRect.Width <= 0 || faceRect.Height <= 0)
                return (false, 0);

            var expanded = ExpandRect(faceRect, bgrImage.Width, bgrImage.Height, 0.4);
            var original = ClampRect(faceRect, bgrImage.Width, bgrImage.Height);

            var exp = Classify(bgrImage, expanded);
            var ori = Classify(bgrImage, original);

            // if both "true" => true, conf=(conf+conf_ori)/2
            if (exp.IsLive && ori.IsLive)
            {
                double avg = (exp.Confidence + ori.Confidence) / 2.0;
                return (true, avg);
            }

            // if expanded true, original false
            if (exp.IsLive && !ori.IsLive)
            {
                // Accept only if expanded is stronger and very confident (>0.95), else reject using false confidence
                if (exp.Confidence > ori.Confidence && exp.Confidence > _strongTrueThreshold)
                    return (true, exp.Confidence);

                return (false, ori.Confidence);
            }

            // if expanded false, original true
            if (!exp.IsLive && ori.IsLive)
            {
                // Accept only if original is stronger and very confident (>0.95), else reject using false confidence
                if (ori.Confidence > exp.Confidence && ori.Confidence > _strongTrueThreshold)
                    return (true, ori.Confidence);

                return (false, exp.Confidence);
            }

            // both false => false, conf=(conf+conf_ori)/2
            {
                double avg = (exp.Confidence + ori.Confidence) / 2.0;
                return (false, avg);
            }
        }

        // Return BOTH predicted class and confidence (max prob), like the C++ code.
        private ClassificationResult Classify(Mat bgrImage, Rect rect)
        {
            if (rect.Width <= 0 || rect.Height <= 0)
                return new ClassificationResult(false, 0);

            using var roi = new Mat(bgrImage, rect).Clone();
            using var resized = new Mat();
            Cv2.Resize(roi, resized, _inputSize);

            using var blob = CvDnn.BlobFromImage(resized, 1.0, _inputSize, _mean, swapRB: false, crop: false);
            _net.SetInput(blob);

            using var prob = _net.Forward();

            // Flatten to 1xN (same idea as prob.reshape(1,1) in C++)
            using var flat = prob.Reshape(1, 1);

            Cv2.MinMaxLoc(flat, out _, out double maxVal, out _, out Point maxLoc);
            int classId = maxLoc.X; // 0 => false, 1 => true (same as C++ classes vector)

            bool isLive = classId == 1;
            return new ClassificationResult(isLive, maxVal);
        }

        private readonly record struct ClassificationResult(bool IsLive, double Confidence);

        private static Rect ExpandRect(Rect rect, int imgW, int imgH, double padFraction)
        {
            int padX = (int)(rect.Width * padFraction);
            int padY = (int)(rect.Height * padFraction);

            int x1 = Math.Max(0, rect.X - padX);
            int y1 = Math.Max(0, rect.Y - padY);
            int x2 = Math.Min(imgW, rect.X + rect.Width + padX);
            int y2 = Math.Min(imgH, rect.Y + rect.Height + padY);

            return new Rect(x1, y1, Math.Max(0, x2 - x1), Math.Max(0, y2 - y1));
        }

        private static Rect ClampRect(Rect rect, int imgW, int imgH)
        {
            int x1 = Math.Max(0, rect.X);
            int y1 = Math.Max(0, rect.Y);
            int x2 = Math.Min(imgW, rect.X + rect.Width);
            int y2 = Math.Min(imgH, rect.Y + rect.Height);

            return new Rect(x1, y1, Math.Max(0, x2 - x1), Math.Max(0, y2 - y1));
        }
    }
}