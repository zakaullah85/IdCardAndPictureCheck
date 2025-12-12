using OpenCvSharp;

namespace IdCardAndPictureCheck
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            using (var matcher = new IdLiveFaceMatcher(
           onnxModelPath: @"models\arcface.onnx",
           haarCascadePath: @"assets\haarcascade_frontalface_default.xml"))
            {

                var result = matcher.MatchIdToCamera(
                     idCardPath: @"D:\images\ID card front.jpeg",
                     cameraImagePath: @"D:\images\client.jpeg",
                     threshold: 0.40 
                 );
                System.IO.File.WriteAllBytes(@"D:\images\annotated.jpg", result.AnnotatedImageBytes);

                MessageBox.Show($"Same person: {result.IsSamePerson}, similarity: {result.BestSimilarity:F3}");
            }

        }

    }
}
