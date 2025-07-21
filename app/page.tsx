"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Loader2, IndianRupee, TrendingUp, MapPin, Building2, Users, Calendar, Briefcase } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface PredictionResult {
  predicted_salary_inr: number
  predicted_salary_formatted: string
  model_name: string
  confidence: string
  r2_score: number
  salary_range: {
    min: string
    max: string
    avg: string
  }
}

export default function SalarySenseApp() {
  const [formData, setFormData] = useState({
    work_year: "2023",
    experience_level: "",
    employment_type: "",
    job_title: "",
    employee_location: "",
    remote_ratio: "",
    company_location: "",
    company_type: "",
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setError("")
  }

  const handlePredict = async () => {
    // Validate form
    const requiredFields = Object.entries(formData).filter(([key, value]) => !value)
    if (requiredFields.length > 0) {
      setError("Please fill in all fields")
      return
    }

    setLoading(true)
    setError("")

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,
          employee_residence: "IN", // Set to India for Indian market
          company_size: formData.company_type === "MNC" ? "L" : formData.company_type === "Startup" ? "S" : "M",
        }),
      })

      if (!response.ok) {
        throw new Error("Prediction failed")
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError("Prediction failed. Please try again.")
      console.error("Prediction error:", err)
    } finally {
      setLoading(false)
    }
  }

  const formatIndianCurrency = (amount: number) => {
    if (amount >= 10000000) {
      return `₹${(amount / 10000000).toFixed(1)} Cr`
    } else if (amount >= 100000) {
      return `₹${(amount / 100000).toFixed(1)} L`
    } else {
      return `₹${amount.toLocaleString("en-IN")}`
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">💰 SalarySense</h1>
          <p className="text-lg text-gray-600">Professional Salary Prediction Platform</p>
          <Badge variant="secondary" className="mt-2">
            Powered by Machine Learning
          </Badge>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Input Form */}
          <div className="xl:col-span-2">
            <Card className="shadow-xl border-0 bg-white/80 backdrop-blur">
              <CardHeader className="bg-gradient-to-r from-orange-500 to-green-500 text-white rounded-t-lg">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Users className="h-6 w-6" />
                  Employee Information
                </CardTitle>
                <CardDescription className="text-orange-100">
                  Enter details to predict salary in Indian Rupees
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label htmlFor="work_year" className="flex items-center gap-2 text-sm font-medium">
                      <Calendar className="h-4 w-4" />
                      Work Year
                    </Label>
                    <Select value={formData.work_year} onValueChange={(value) => handleInputChange("work_year", value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select year" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2020">2020</SelectItem>
                        <SelectItem value="2021">2021</SelectItem>
                        <SelectItem value="2022">2022</SelectItem>
                        <SelectItem value="2023">2023</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="experience_level" className="flex items-center gap-2 text-sm font-medium">
                      <TrendingUp className="h-4 w-4" />
                      Experience Level
                    </Label>
                    <Select
                      value={formData.experience_level}
                      onValueChange={(value) => handleInputChange("experience_level", value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select experience" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Fresher">Fresher (0-1 years)</SelectItem>
                        <SelectItem value="Junior">Junior (1-3 years)</SelectItem>
                        <SelectItem value="Senior">Senior (3-7 years)</SelectItem>
                        <SelectItem value="Manager">Manager (7+ years)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div>
                  <Label htmlFor="job_title" className="flex items-center gap-2 text-sm font-medium">
                    <Briefcase className="h-4 w-4" />
                    Job Title
                  </Label>
                  <Select value={formData.job_title} onValueChange={(value) => handleInputChange("job_title", value)}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select job title" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Data Scientist">Data Scientist</SelectItem>
                      <SelectItem value="Data Engineer">Data Engineer</SelectItem>
                      <SelectItem value="Data Analyst">Data Analyst</SelectItem>
                      <SelectItem value="ML Engineer">ML Engineer</SelectItem>
                      <SelectItem value="Business Analyst">Business Analyst</SelectItem>
                      <SelectItem value="Analytics Manager">Analytics Manager</SelectItem>
                      <SelectItem value="Data Architect">Data Architect</SelectItem>
                      <SelectItem value="Research Scientist">Research Scientist</SelectItem>
                      <SelectItem value="AI Engineer">AI Engineer</SelectItem>
                      <SelectItem value="Product Analyst">Product Analyst</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="employment_type" className="flex items-center gap-2 text-sm font-medium">
                    <Building2 className="h-4 w-4" />
                    Employment Type
                  </Label>
                  <Select
                    value={formData.employment_type}
                    onValueChange={(value) => handleInputChange("employment_type", value)}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select employment type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Full_Time">Full Time</SelectItem>
                      <SelectItem value="Part_Time">Part Time</SelectItem>
                      <SelectItem value="Contract">Contract</SelectItem>
                      <SelectItem value="Freelance">Freelance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label htmlFor="employee_location" className="flex items-center gap-2 text-sm font-medium">
                      <MapPin className="h-4 w-4" />
                      Employee Location
                    </Label>
                    <Select
                      value={formData.employee_location}
                      onValueChange={(value) => handleInputChange("employee_location", value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select city" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Bangalore">Bangalore</SelectItem>
                        <SelectItem value="Mumbai">Mumbai</SelectItem>
                        <SelectItem value="Delhi">Delhi</SelectItem>
                        <SelectItem value="Hyderabad">Hyderabad</SelectItem>
                        <SelectItem value="Pune">Pune</SelectItem>
                        <SelectItem value="Chennai">Chennai</SelectItem>
                        <SelectItem value="Kolkata">Kolkata</SelectItem>
                        <SelectItem value="Gurgaon">Gurgaon</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="company_location" className="flex items-center gap-2 text-sm font-medium">
                      <Building2 className="h-4 w-4" />
                      Company Location
                    </Label>
                    <Select
                      value={formData.company_location}
                      onValueChange={(value) => handleInputChange("company_location", value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select city" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Bangalore">Bangalore</SelectItem>
                        <SelectItem value="Mumbai">Mumbai</SelectItem>
                        <SelectItem value="Delhi">Delhi</SelectItem>
                        <SelectItem value="Hyderabad">Hyderabad</SelectItem>
                        <SelectItem value="Pune">Pune</SelectItem>
                        <SelectItem value="Chennai">Chennai</SelectItem>
                        <SelectItem value="Kolkata">Kolkata</SelectItem>
                        <SelectItem value="Gurgaon">Gurgaon</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label htmlFor="remote_ratio" className="flex items-center gap-2 text-sm font-medium">
                      <MapPin className="h-4 w-4" />
                      Remote Work Ratio
                    </Label>
                    <Select
                      value={formData.remote_ratio}
                      onValueChange={(value) => handleInputChange("remote_ratio", value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select remote ratio" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">0% - Office Only</SelectItem>
                        <SelectItem value="50">50% - Hybrid</SelectItem>
                        <SelectItem value="100">100% - Fully Remote</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="company_type" className="flex items-center gap-2 text-sm font-medium">
                      <Building2 className="h-4 w-4" />
                      Company Type
                    </Label>
                    <Select
                      value={formData.company_type}
                      onValueChange={(value) => handleInputChange("company_type", value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select company type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="MNC">MNC</SelectItem>
                        <SelectItem value="Startup">Startup</SelectItem>
                        <SelectItem value="Government">Government</SelectItem>
                        <SelectItem value="Corporate">Corporate</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <Button
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-orange-600 to-green-600 hover:from-orange-700 hover:to-green-700 text-white py-3 text-lg font-semibold"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <IndianRupee className="mr-2 h-5 w-5" />
                      Predict Salary
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="xl:col-span-1">
            <Card className="shadow-xl border-0 bg-white/80 backdrop-blur h-fit">
              <CardHeader className="bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-t-lg">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <TrendingUp className="h-6 w-6" />
                  Prediction Results
                </CardTitle>
                <CardDescription className="text-green-100">AI-powered salary prediction in INR</CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                {prediction ? (
                  <div className="space-y-6">
                    {/* Main Prediction */}
                    <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border-2 border-green-200">
                      <div className="text-4xl font-bold text-green-700 mb-2">
                        {prediction.predicted_salary_formatted}
                      </div>
                      <p className="text-green-600 font-medium">Predicted Annual Salary</p>
                      <Badge variant="outline" className="mt-3 border-green-300 text-green-700 bg-green-50">
                        {prediction.confidence} Confidence
                      </Badge>
                    </div>

                    {/* Model Info */}
                    <div className="grid grid-cols-1 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="text-lg font-semibold text-blue-700">{prediction.model_name}</div>
                        <p className="text-sm text-blue-600">ML Model Used</p>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <div className="text-lg font-semibold text-purple-700">
                          {(prediction.r2_score * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-purple-600">Model Accuracy</p>
                      </div>
                    </div>

                    {/* Salary Range Context */}
                    <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                      <h4 className="font-semibold mb-3 flex items-center gap-2 text-orange-800">
                        <IndianRupee className="h-4 w-4" />
                        Salary Context
                      </h4>
                      <div className="space-y-2 text-sm text-orange-700">
                        <div className="flex justify-between">
                          <span>Industry Average:</span>
                          <span className="font-medium">{prediction.salary_range.avg}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Range:</span>
                          <span className="font-medium">
                            {prediction.salary_range.min} - {prediction.salary_range.max}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Additional Info */}
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                      <h4 className="font-semibold mb-2 text-gray-800">Important Notes</h4>
                      <div className="text-xs text-gray-600 space-y-1">
                        <p>• Based on current market trends and data</p>
                        <p>• Actual salaries may vary by company and location</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <IndianRupee className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p className="text-lg">Fill the form to predict salary</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Built with Next.js, Python, and Machine Learning • SalarySense 2024</p>
          <p className="mt-1">Empowering Data Science Professionals</p>
        </div>
      </div>
    </div>
  )
}
