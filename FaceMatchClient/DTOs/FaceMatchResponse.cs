

namespace FaceMatchClient.DTOs
{
    public class FaceMatchResponseDto
    {
        public bool IsSamePerson { get; set; }
        public double BestSimilarity { get; set; }
        public FaceRect MatchedFaceRect { get; set; }
        public string AnnotatedImageBase64 { get; set; }
        public string Error { get; set; }
    }
}
