using OpenCvSharp;

public class FaceMatchResult
{
    public bool IsSamePerson { get; set; }
    public double BestSimilarity { get; set; }
    public Rect MatchedFaceRect { get; set; }
    public bool IsLiveFace { get; set; }
    public double AntiSpoofConfidence { get; set; }

    /// <summary>
    /// JPEG-encoded annotated camera image.
    /// </summary>
    public byte[] AnnotatedImageBytes { get; set; }
}
