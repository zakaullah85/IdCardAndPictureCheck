using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Size = OpenCvSharp.Size;

public class ArcFaceEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly CascadeClassifier _faceCascade;
    private readonly Size _inputSize = new Size(112, 112); // ArcFace expected size
    private readonly string _inputName;
    private readonly string _outputName;

    // Adjust default paths as needed
    public ArcFaceEmbedder(
        string onnxModelPath = @"models\arcface.onnx",
        string haarCascadePath = @"assets\haarcascade_frontalface_default.xml")
    {
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException("ArcFace ONNX model not found.", onnxModelPath);

        if (!File.Exists(haarCascadePath))
            throw new FileNotFoundException("Haar cascade not found.", haarCascadePath);

        _session = new InferenceSession(onnxModelPath);
        _faceCascade = new CascadeClassifier(haarCascadePath);

        // Use real names from model so we don't hardcode "data" / "fc1"
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    public void Dispose()
    {
        _session?.Dispose();
        _faceCascade?.Dispose();
    }

    /// <summary>
    /// Extracts the largest face from an image file and returns its 512-D embedding.
    /// </summary>
    public float[] GetEmbeddingFromFile(string imagePath)
    {
        using var img = Cv2.ImRead(imagePath);
        if (img.Empty())
            throw new InvalidOperationException("Unable to read image: " + imagePath);

        using var rgb = new Mat();
        Cv2.CvtColor(img, rgb, ColorConversionCodes.BGR2RGB);

        var faceRect = DetectLargestFace(rgb);
        if (faceRect.Width <= 0 || faceRect.Height <= 0)
            throw new Exception("No face detected in image: " + imagePath);

        var padded = PadRect(faceRect, rgb.Width, rgb.Height, 0.2);

        using var faceRoi = new Mat(rgb, padded).Clone();
        using var resized = new Mat();
        Cv2.Resize(faceRoi, resized, _inputSize);

        return RunArcFace(resized);
    }

    /// <summary>
    /// Returns (faceRect, embedding) for all faces in an image.
    /// </summary>
    public (Rect rect, float[] embedding)[] GetAllFaceEmbeddings(string imagePath)
    {
        using var img = Cv2.ImRead(imagePath);
        if (img.Empty())
            throw new InvalidOperationException("Unable to read image: " + imagePath);

        using var rgb = new Mat();
        Cv2.CvtColor(img, rgb, ColorConversionCodes.BGR2RGB);

        var faces = DetectFaces(rgb);
        if (faces.Length == 0)
            return Array.Empty<(Rect, float[])>();

        return faces.Select(rect =>
        {
            var padded = PadRect(rect, rgb.Width, rgb.Height, 0.2);
            using var roi = new Mat(rgb, padded).Clone();
            using var resized = new Mat();
            Cv2.Resize(roi, resized, _inputSize);

            var emb = RunArcFace(resized);
            return (padded, emb);
        }).ToArray();
    }

    private Rect DetectLargestFace(Mat rgb)
    {
        using var gray = new Mat();
        Cv2.CvtColor(rgb, gray, ColorConversionCodes.RGB2GRAY);
        Cv2.EqualizeHist(gray, gray);

        var faces = _faceCascade.DetectMultiScale(
            gray,
            scaleFactor: 1.1,
            minNeighbors: 4,
            flags: 0,
            minSize: new Size(60, 60));

        if (faces == null || faces.Length == 0)
            return new Rect(0, 0, 0, 0);

        return faces.OrderByDescending(r => r.Width * r.Height).First();
    }

    private Rect[] DetectFaces(Mat rgb)
    {
        using var gray = new Mat();
        Cv2.CvtColor(rgb, gray, ColorConversionCodes.RGB2GRAY);
        Cv2.EqualizeHist(gray, gray);

        var faces = _faceCascade.DetectMultiScale(
            gray,
            scaleFactor: 1.1,
            minNeighbors: 4,
            flags: 0,
            minSize: new Size(60, 60));

        return faces ?? Array.Empty<Rect>();
    }

    private static Rect PadRect(Rect r, int imgW, int imgH, double padFraction)
    {
        int padX = (int)(r.Width * padFraction);
        int padY = (int)(r.Height * padFraction);

        int x = Math.Max(0, r.X - padX);
        int y = Math.Max(0, r.Y - padY);
        int x2 = Math.Min(imgW, r.X + r.Width + padX);
        int y2 = Math.Min(imgH, r.Y + r.Height + padY);

        return new Rect(x, y, x2 - x, y2 - y);
    }

    /// <summary>
    /// Run ArcFace ONNX model (expects RGB 112x112, normalized, NHWC [1,112,112,3]).
    /// </summary>
    private float[] RunArcFace(Mat faceRgb112)
    {
        // Input tensor: [1, 112, 112, 3] (NHWC)
        var input = new DenseTensor<float>(new[] { 1, _inputSize.Height, _inputSize.Width, 3 });

        for (int y = 0; y < _inputSize.Height; y++)
        {
            for (int x = 0; x < _inputSize.Width; x++)
            {
                var pixel = faceRgb112.At<Vec3b>(y, x); // RGB because we converted before
                // Same normalization as the model card example: (img - 127.5) / 128.0 
                input[0, y, x, 0] = (pixel.Item0 - 127.5f) / 128.0f; // R
                input[0, y, x, 1] = (pixel.Item1 - 127.5f) / 128.0f; // G
                input[0, y, x, 2] = (pixel.Item2 - 127.5f) / 128.0f; // B
            }
        }

        var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, input) };
        using var results = _session.Run(inputs);

        var outputTensor = results.First(r => r.Name == _outputName)
                                     .AsEnumerable<float>()
                                     .ToArray();

        // L2-normalize embedding
        float norm = (float)Math.Sqrt(outputTensor.Sum(v => v * v));
        if (norm > 0)
        {
            for (int i = 0; i < outputTensor.Length; i++)
                outputTensor[i] /= norm;
        }

        return outputTensor;
    }

    public static double CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Embedding sizes do not match.");

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
