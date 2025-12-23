using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Dnn;

public sealed class AntiSpoofingDetector : IDisposable
{
    private readonly Net _net;
    private readonly Size _inputSize = new Size(227, 227);
    private readonly Scalar _mean = new Scalar(90.0, 198.0, 121.0);

    public AntiSpoofingDetector(string prototxtPath, string caffemodelPath)
    {
        if (!File.Exists(prototxtPath))
            throw new FileNotFoundException("Anti-spoofing prototxt not found.", prototxtPath);
        if (!File.Exists(caffemodelPath))
            throw new FileNotFoundException("Anti-spoofing caffemodel not found.", caffemodelPath);

        _net = CvDnn.ReadNetFromCaffe(prototxtPath, caffemodelPath);
    }

    public void Dispose()
    {
        _net?.Dispose();
    }

    public (bool isLive, double confidence) Evaluate(Mat bgrImage, Rect faceRect)
    {
        if (bgrImage.Empty() || faceRect.Width <= 0 || faceRect.Height <= 0)
            return (false, 0);

        var expanded = ExpandRect(faceRect, bgrImage.Width, bgrImage.Height, 0.4);
        var clamped = ClampRect(faceRect, bgrImage.Width, bgrImage.Height);

        var (liveExpanded, confExpanded) = Classify(bgrImage, expanded);
        var (liveOriginal, confOriginal) = Classify(bgrImage, clamped);

        if (liveExpanded && liveOriginal)
            return (true, (confExpanded + confOriginal) / 2.0);

        if (liveExpanded && !liveOriginal)
        {
            if (confExpanded > confOriginal && confExpanded > 0.95)
                return (true, confExpanded);
            return (false, confOriginal);
        }

        if (!liveExpanded && liveOriginal)
        {
            if (confOriginal > confExpanded && confOriginal > 0.95)
                return (true, confOriginal);
            return (false, confExpanded);
        }

        return (false, (confExpanded + confOriginal) / 2.0);
    }

    private (bool isLive, double confidence) Classify(Mat bgrImage, Rect rect)
    {
        if (rect.Width <= 0 || rect.Height <= 0)
            return (false, 0);

        using var roi = new Mat(bgrImage, rect).Clone();
        using var resized = new Mat();
        Cv2.Resize(roi, resized, _inputSize);

        using var blob = CvDnn.BlobFromImage(resized, 1.0, _inputSize, _mean, swapRB: false, crop: false);
        _net.SetInput(blob);
        using var prob = _net.Forward();

        using var flat = prob.Reshape(1, 1);
        Cv2.MinMaxLoc(flat, out _, out double maxVal, out _, out Point maxLoc);

        int classId = maxLoc.X;
        bool isLive = classId == 1;
        return (isLive, maxVal);
    }

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
