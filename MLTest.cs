using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace MLTest
{
    class Program
    {
        // STEP 1: Define data structures
        public class ClaimsData
        {
            [Column("0")]
            public string InceptionDate;

            [Column("1")]
            public string ExpirationDate;

            [Column("2")]
            public string PolicyStatus;

            [Column("3")]
            public int PoliciesPerDocument;

            [Column("4")]
            public string NewRenewal;

            [Column("5")]
            public string BusinessTypeCode;

            [Column("6")]
            public string PostalCode;

            [Column("7")]
            public float ThreeYearClaims;

            [Column("8")]
            public float Premium;
        }
        public class RatePrediction
        {
            [ColumnName("Score")]
            public float PredictedRate;
        }

        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load  data
            var pipeline = new LearningPipeline();

            string dataPath = "./claims.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<ClaimsData>(separator: '\t'));

            // STEP 3: Transform your data
            pipeline.Add(new Dictionarizer("InceptionDate"));
            pipeline.Add(new Dictionarizer("ExpirationDate"));
            pipeline.Add(new Dictionarizer("NewRenewal"));
            pipeline.Add(new Dictionarizer("BusinessTypeCode"));
            pipeline.Add(new Dictionarizer("PostalCode"));


            pipeline.Add(new ColumnCopier("PoliciesPerDocument"));          
            pipeline.Add(new ColumnCopier(("Premium", "Label")));

            // Puts all features into a vector
            pipeline.Add(new CategoricalOneHotVectorizer("InceptionDate", "ExpirationDate", "PoliciesPerDocument", "NewRenewal", "BusinessTypeCode", "PostalCode", "ThreeYearClaims"));
            pipeline.Add(new ColumnConcatenator("Features", "InceptionDate", "ExpirationDate", "PoliciesPerDocument", "NewRenewal", "BusinessTypeCode", "PostalCode", "ThreeYearClaims"));

            // STEP 4: Add learner; I've attached several that are valid for this data set for people
            // to play with, just uncomment one and let it run

            pipeline.Add(new FastTreeRegressor());  
            //pipeline.Add( new PoissonRegressor{MaxIterations=1000,L1Weight=0.8f,L2Weight=0.2f} );
            //pipeline.Add( new FastTreeTweedieRegressor());
            //pipeline.Add( new FastForestRegressor{ NumTrees = 1000, NumLeaves = 500, NumThreads = 10, EntropyCoefficient = 0.3 });
            //pipeline.Add( new FastTreeRegressor{ NumTrees = 200,LearningRates = 0.4f, DropoutRate = 0.01f });
            //WARNING: This one is a hog and takes forever
            //pipeline.Add( new GeneralizedAdditiveModelRegressor());
            //pipeline.Add( new StochasticDualCoordinateAscentRegressor{ MaxIterations = 1000, NumThreads = 5 });

            // STEP 5: Train model based on the data set
            var model = pipeline.Train<ClaimsData, RatePrediction>();

            // STEP 6: Evaluate the model
            var testData = new TextLoader(dataPath).CreateFrom<ClaimsData>(separator: '\t');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"****EVALUATION****\nRms (lower is better) = {metrics.Rms}");
            Console.WriteLine($"RSquared (closer to 1.0 is better)= {metrics.RSquared}");

            // STEP 7: Use it
            var prediction = model.Predict(new ClaimsData()
            {
                InceptionDate = "9/10/2018",
                ExpirationDate = "9/10/209",
                PolicyStatus = "Firm Order",
                PoliciesPerDocument = 1,
                NewRenewal = "New",
                BusinessTypeCode = "VVC",
                PostalCode = "MK46 5JA"
            });

            Console.WriteLine($"****PREDICTION****\nPredicted rate is: {prediction.PredictedRate}");
        }
    }
}