using System;
using OpenCvSharp;

namespace IDCardFaceMatchHelper64
{
    internal static class IdFacePreprocessor
    {
        public static Mat ApplyClahe(Mat bgr)
        {
            using var lab = new Mat();
            Cv2.CvtColor(bgr, lab, ColorConversionCodes.BGR2Lab);

            Cv2.Split(lab, out Mat[] ch); // L, A, B
            try
            {
                using var clahe = Cv2.CreateCLAHE(clipLimit: 2.5, tileGridSize: new Size(8, 8));
                using var l2 = new Mat();
                clahe.Apply(ch[0], l2);
                l2.CopyTo(ch[0]);

                using var merged = new Mat();
                Cv2.Merge(ch, merged);

                var outBgr = new Mat();
                Cv2.CvtColor(merged, outBgr, ColorConversionCodes.Lab2BGR);
                return outBgr;
            }
            finally
            {
                foreach (var m in ch) m.Dispose();
            }
        }

        public static Mat ApplyGamma(Mat bgr, double gamma)
        {
            var lut = new Mat(1, 256, MatType.CV_8UC1);
            for (int i = 0; i < 256; i++)
            {
                double v = Math.Pow(i / 255.0, 1.0 / gamma) * 255.0;
                lut.Set(0, i, (byte)Math.Clamp((int)Math.Round(v), 0, 255));
            }

            var outImg = new Mat();
            Cv2.LUT(bgr, lut, outImg);
            lut.Dispose();
            return outImg;
        }

        /// <summary>
        /// Downsample->Upsample + gentle denoise to suppress high-frequency guilloché/security lines.
        /// </summary>
        public static Mat RemoveMoiré(Mat bgr)
        {
            using var small = new Mat();
            Cv2.Resize(bgr, small, new Size(), 0.5, 0.5, InterpolationFlags.Area);

            using var restored = new Mat();
            Cv2.Resize(small, restored, bgr.Size(), 0, 0, InterpolationFlags.Cubic);

            var den = new Mat();
            Cv2.FastNlMeansDenoisingColored(restored, den, h: 6, hColor: 6, templateWindowSize: 7, searchWindowSize: 21);

            var outImg = ApplyClahe(den);
            den.Dispose();
            return outImg;
        }

        /// <summary>
        /// Soft elliptical mask to suppress background / border texture around face.
        /// </summary>
        public static Mat ApplySoftFaceMask(Mat bgr)
        {
            using var mask = new Mat(bgr.Rows, bgr.Cols, MatType.CV_8UC1, Scalar.All(0));

            var center = new Point(bgr.Cols / 2, bgr.Rows / 2);
            var axes = new Size((int)(bgr.Cols * 0.42), (int)(bgr.Rows * 0.48));
            Cv2.Ellipse(mask, center, axes, 0, 0, 360, Scalar.All(255), -1);

            Cv2.GaussianBlur(mask, mask, new Size(31, 31), 0);

            using var bg = new Mat(bgr.Size(), bgr.Type(), new Scalar(127, 127, 127));

            using var bgrF = new Mat(); bgr.ConvertTo(bgrF, MatType.CV_32FC3);
            using var bgF = new Mat(); bg.ConvertTo(bgF, MatType.CV_32FC3);
            using var mF = new Mat(); mask.ConvertTo(mF, MatType.CV_32FC1, 1.0 / 255.0);

            using var m3 = new Mat();
            Cv2.Merge(new[] { mF, mF, mF }, m3);

            using var inv = new Mat();
            Cv2.Subtract(Scalar.All(1.0), m3, inv);

            using var fgPart = new Mat();
            using var bgPart = new Mat();
            Cv2.Multiply(bgrF, m3, fgPart);
            Cv2.Multiply(bgF, inv, bgPart);

            using var sum = new Mat();
            Cv2.Add(fgPart, bgPart, sum);

            var outImg = new Mat();
            sum.ConvertTo(outImg, MatType.CV_8UC3);
            return outImg;
        }

        /// <summary>
        /// Build robust variants for ID portrait (best-of-N).
        /// Caller must Dispose all returned mats.
        /// </summary>
        public static Mat[] BuildIdFaceVariants(Mat faceBgr)
        {
            // A: CLAHE only
            var v1 = ApplyClahe(faceBgr);

            // B: Gamma + CLAHE (helps faded scans)
            var g = ApplyGamma(faceBgr, 0.85);
            var v2 = ApplyClahe(g);
            g.Dispose();

            // C: Moire removal (down/up + NLM + CLAHE)
            var v3 = RemoveMoiré(faceBgr);

            // D: Soft mask + CLAHE
            var masked = ApplySoftFaceMask(faceBgr);
            var v4 = ApplyClahe(masked);
            masked.Dispose();

            return new[] { v1, v2, v3, v4 };
        }
    }
}
