import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Layout } from "@/components/Layout";
import { Brain, Zap, Shield, TrendingUp, Image, BarChart3 } from "lucide-react";
import { apiService } from "@/lib/api";
import { useEffect, useState } from "react";

const features = [
  {
    icon: Image,
    title: "Single Predictions",
    description:
      "Upload individual images for instant ML-powered analysis with confidence scores.",
  },
  {
    icon: BarChart3,
    title: "Batch Processing",
    description:
      "Process multiple images at once for efficient bulk predictions.",
  },
  {
    icon: TrendingUp,
    title: "Prediction History",
    description:
      "Track and analyze your prediction history with detailed insights.",
  },
  {
    icon: Zap,
    title: "Real-time Results",
    description:
      "Get instant predictions with optional heatmap visualizations.",
  },
  {
    icon: Shield,
    title: "Secure Authentication",
    description: "JWT-based authentication ensures your data stays protected.",
  },
  {
    icon: Brain,
    title: "Advanced ML Models",
    description: "Powered by state-of-the-art machine learning algorithms.",
  },
];

export default function Index() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    setIsAuthenticated(apiService.isAuthenticated());
  }, []);

  return (
    <Layout>
      {/* Hero Section */}
      <section className="text-center py-16 space-y-8">
        <div className="space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold">
            Machine Learning
            <span className="block mt-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Prediction API
            </span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Professional ML prediction service with real-time analysis, batch
            processing, and comprehensive history tracking.
          </p>
        </div>

        <div className="flex flex-wrap gap-4 justify-center">
          {!isAuthenticated ? (
            <>
              <Link to="/login">
                <Button
                  variant="default"
                  size="lg"
                  className="gap-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-sm"
                >
                  Get Started
                  <Brain className="h-5 w-5" />
                </Button>
              </Link>
              <Link to="/register">
                <Button variant="outline" size="lg">
                  Sign Up Free
                </Button>
              </Link>
            </>
          ) : (
            <>
              <Link to="/dashboard">
                <Button
                  variant="default"
                  size="lg"
                  className="gap-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-sm"
                >
                  Go to Dashboard
                  <TrendingUp className="h-5 w-5" />
                </Button>
              </Link>
              <Link to="/predict">
                <Button variant="outline" size="lg" className="gap-2">
                  Make Prediction
                  <Image className="h-5 w-5" />
                </Button>
              </Link>
            </>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto pt-12">
          <Card className="shadow-md border-0">
            <CardContent className="pt-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">99.2%</div>
              <div className="text-sm text-muted-foreground">Accuracy Rate</div>
            </CardContent>
          </Card>
          <Card className="shadow-md border-0">
            <CardContent className="pt-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                &lt;500ms
              </div>
              <div className="text-sm text-muted-foreground">Response Time</div>
            </CardContent>
          </Card>
          <Card className="shadow-md border-0">
            <CardContent className="pt-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">24/7</div>
              <div className="text-sm text-muted-foreground">Availability</div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Powerful Features</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Everything you need for professional machine learning predictions
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card
                key={index}
                className="shadow-md hover:shadow-lg transition-all duration-300 border-0"
              >
                <CardContent className="pt-6">
                  <div className="rounded-lg bg-blue-50 w-12 h-12 flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 text-center">
        <Card className="bg-gradient-to-r from-blue-500 to-purple-600 text-white border-0 shadow-xl">
          <CardContent className="py-12">
            <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
            <p className="text-lg mb-8 opacity-90 max-w-2xl mx-auto">
              Join thousands of developers using our ML prediction API for their
              applications.
            </p>
            {!isAuthenticated ? (
              <Link to="/register">
                <Button
                  variant="secondary"
                  size="lg"
                  className="bg-white text-blue-600 hover:bg-gray-100 shadow-sm"
                >
                  Start Predicting Now
                </Button>
              </Link>
            ) : (
              <Link to="/predict">
                <Button
                  variant="secondary"
                  size="lg"
                  className="bg-white text-blue-600 hover:bg-gray-100 shadow-sm"
                >
                  Make Your First Prediction
                </Button>
              </Link>
            )}
          </CardContent>
        </Card>
      </section>
    </Layout>
  );
}
