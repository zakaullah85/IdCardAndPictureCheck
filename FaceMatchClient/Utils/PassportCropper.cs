using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace FaceMatchClient.Utils
{


    public static class PassportCropper
    {
        public static readonly Size PassportSize = new Size(413, 531);

        public static Bitmap CropPassport(
            string originalImagePath,
            Rectangle faceRect,
            Size? outputSize = null)
        {
            if (faceRect.Width <= 0 || faceRect.Height <= 0)
                throw new ArgumentException("Face rectangle is empty.", nameof(faceRect));

            using var srcImg = new Bitmap(originalImagePath);

            var targetSize = outputSize ?? PassportSize;
            double desiredRatio = targetSize.Width / (double)targetSize.Height;

            // 1. Padded region around the face
            int padX = (int)(faceRect.Width * 0.1);
            int padTop = (int)(faceRect.Height * 0.1);
            int padBottom = (int)(faceRect.Height * 0.1);

            int x = Math.Max(0, faceRect.X - padX);
            int y = Math.Max(0, faceRect.Y - padTop);
            int x2 = Math.Min(srcImg.Width, faceRect.Right + padX);
            int y2 = Math.Min(srcImg.Height, faceRect.Bottom + padBottom);

            int cropW = x2 - x;
            int cropH = y2 - y;
            if (cropW <= 0 || cropH <= 0)
                throw new InvalidOperationException("Calculated crop rectangle is invalid.");

            // 2. Adjust to desired aspect ratio
            double currentRatio = cropW / (double)cropH;

            if (currentRatio > desiredRatio)
            {
                int newH = (int)(cropW / desiredRatio);
                int diff = newH - cropH;
                y -= diff / 2;
                if (y < 0) y = 0;
                if (y + newH > srcImg.Height) newH = srcImg.Height - y;
                cropH = newH;
            }
            else
            {
                int newW = (int)(cropH * desiredRatio);
                int diff = newW - cropW;
                x -= diff / 2;
                if (x < 0) x = 0;
                if (x + newW > srcImg.Width) newW = srcImg.Width - x;
                cropW = newW;
            }

            var cropRect = new Rectangle(x, y, cropW, cropH);

            // 3. First: just crop (no resize)
            var cropped = srcImg.Clone(cropRect, srcImg.PixelFormat);

            // If you don't want a fixed passport size, just return the cropped bitmap here:
            // return cropped;

            // 4. Resize cropped region to final passport size
            var passportBmp = new Bitmap(targetSize.Width, targetSize.Height, PixelFormat.Format24bppRgb);

            using (var g = Graphics.FromImage(passportBmp))
            {
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                g.DrawImage(
                    cropped,
                    destRect: new Rectangle(0, 0, passportBmp.Width, passportBmp.Height),
                    srcX: 0,
                    srcY: 0,
                    srcWidth: cropped.Width,
                    srcHeight: cropped.Height,
                    srcUnit: GraphicsUnit.Pixel);
            }

            cropped.Dispose();
            return passportBmp;
        }
    }


}
