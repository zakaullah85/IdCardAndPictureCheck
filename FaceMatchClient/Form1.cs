using FaceMatchClient.Utils;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace FaceMatchClient
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            button1.Enabled = false;
            string helperExe = @"C:\Users\zakau\source\repos\IdCardAndPictureCheck\IDCardFaceMatchHelper64\bin\Release\net9.0\publish\win-x64\IDCardFaceMatchHelper64.exe";
            string idCardPath = @"D:\images\ID-Card1.jpeg";
            string cameraPath = @"D:\images\client4.jpeg";
            pictureBox2.ImageLocation = cameraPath;
            pictureBox3.ImageLocation = idCardPath;
            double faceScaleFactor = 0.9;
            double threshold = 0.5;
            double liveThreshold = 0.95;
        retryFaceMatch:
            var resp = await FaceMatchService.RunFaceMatchAsync(helperExe, idCardPath, cameraPath,faceScaleFactor,threshold,liveThreshold);


            var r = resp.MatchedFaceRect;
            if (r == null || r.Width <= 0 || r.Height <= 0)
            {
                if (faceScaleFactor < 1.5)
                {
                    faceScaleFactor = faceScaleFactor + 0.1;
                    goto retryFaceMatch;
                }
                MessageBox.Show("No matching face rectangle returned.");

            }
            if (faceScaleFactor < 1.5 && !resp.IsSamePerson)
            {
                faceScaleFactor = faceScaleFactor + 0.1;
                goto retryFaceMatch;
            }
            if (resp.IsSamePerson)
            {



                Rectangle faceRect = new Rectangle(r.X, r.Y, r.Width, r.Height);

                // Dispose previous image to avoid leaks
                if (pictureBox1.Image != null)
                    pictureBox1.Image.Dispose();

                // Crop at default passport size (or null to keep cropped size)
                var passport = PassportCropper.CropPassport(cameraPath, faceRect);
                // Or: var passport = PassportCropper.CropPassport(cameraPath, faceRect, null); // no resize

                pictureBox1.SizeMode = PictureBoxSizeMode.CenterImage;

                // Optionally, set PictureBox size closer to the passport bitmap size
                pictureBox1.Width = passport.Width;
                pictureBox1.Height = passport.Height;

                pictureBox1.Image = passport;




            }
            else
            {
                pictureBox1.Image = null;
            }
            // MessageBox.Show($"Same person: {resp.IsSamePerson}, similarity: {resp.BestSimilarity:F3}");
            button1.Enabled = true;
            //  // Show annotated image in UI
            if (resp.AnnotatedImageBase64 != null)
            {
                var bmp = Utils.ImageConverter.Base64ToBitmap(resp.AnnotatedImageBase64);
                bmp.Save(@"D:\images\annotated_from_client.jpg");
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}
