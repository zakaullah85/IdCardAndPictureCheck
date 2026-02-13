using System;
using System.IO;
using System.Text.Json;

namespace IDCardFaceMatchHelper64
{
    public class FaceRect
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class FaceMatchResponse
    {
        public bool IsSamePerson { get; set; }
        public double BestSimilarity { get; set; }
        public FaceRect MatchedFaceRect { get; set; }
        public bool IsLiveFace { get; set; }
        public double AntiSpoofConfidence { get; set; }
        public string AnnotatedImageBase64 { get; set; }
        public string Error { get; set; }
    }

    internal class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 5)
            {
                var errorResp = new FaceMatchResponse
                {
                    IsSamePerson = false,
                    BestSimilarity = 0,
                    MatchedFaceRect = null,
                    AnnotatedImageBase64 = null,
                    Error = "Usage: IDCardFaceMatchHelper64 <idCardPath> <cameraImagePath> <threshold> <faceConfThreshold> <liveThreshold>"
                };

                Console.WriteLine(JsonSerializer.Serialize(errorResp, FaceMatchJsonContext.Default.FaceMatchResponse));
                return 1;
            }

            string idCardPath = args[0];
            string cameraImagePath = args[1];
            double threshold = double.Parse(args[2]);
            double faceConfThreshold = double.Parse(args[3]);
            double liveThreshold = double.Parse(args[4]);

            try
            {
                if (!File.Exists(idCardPath))
                    throw new FileNotFoundException("ID card image not found.", idCardPath);

                if (!File.Exists(cameraImagePath))
                    throw new FileNotFoundException("Camera image not found.", cameraImagePath);

                using var matcher = new IdLiveFaceMatcher(faceConfThreshold: faceConfThreshold, liveThreshold: liveThreshold);

                var result = matcher.MatchIdToCamera(idCardPath, cameraImagePath, threshold);

                string base64 = result.AnnotatedImageBytes != null
                    ? Convert.ToBase64String(result.AnnotatedImageBytes)
                    : null;

                var rect = result.MatchedFaceRect;
                FaceRect rectDto = (rect.Width > 0 && rect.Height > 0)
                    ? new FaceRect { X = rect.X, Y = rect.Y, Width = rect.Width, Height = rect.Height }
                    : null;

                var resp = new FaceMatchResponse
                {
                    IsSamePerson = result.IsSamePerson,
                    BestSimilarity = result.BestSimilarity,
                    MatchedFaceRect = rectDto,
                    IsLiveFace = result.IsLiveFace,
                    AntiSpoofConfidence = result.AntiSpoofConfidence,
                    AnnotatedImageBase64 = base64,
                    Error = null
                };

                Console.WriteLine(JsonSerializer.Serialize(resp, FaceMatchJsonContext.Default.FaceMatchResponse));
                return 0;
            }
            catch (Exception ex)
            {
                var errorResp = new FaceMatchResponse
                {
                    IsSamePerson = false,
                    BestSimilarity = 0,
                    MatchedFaceRect = null,
                    AnnotatedImageBase64 = null,
                    Error = ex.Message
                };

                Console.WriteLine(JsonSerializer.Serialize(errorResp, FaceMatchJsonContext.Default.FaceMatchResponse));
                return 2;
            }
        }
    }
}