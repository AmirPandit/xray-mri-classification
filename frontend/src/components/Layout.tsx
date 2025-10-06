import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Home,
  Activity,
  Image,
  Images,
  History,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { apiService } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dashboard", href: "/dashboard", icon: Activity },
  { name: "Single Prediction", href: "/predict", icon: Image },
  { name: "Batch Prediction", href: "/batch", icon: Images },
  { name: "History", href: "/history", icon: History },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Check authentication status
  useEffect(() => {
    setIsAuthenticated(apiService.isAuthenticated());
  }, [location.pathname]); // Re-check when route changes

  const handleLogout = () => {
    apiService.logout();
    setIsAuthenticated(false);
    setMobileMenuOpen(false);

    localStorage.clear();
    sessionStorage.clear();

    toast({
      title: "Logged out successfully",
      description: "You have been logged out.",
    });
    navigate("/login");
  };

  const isActiveRoute = (href: string) => {
    if (href === "/") {
      return location.pathname === "/";
    }
    return location.pathname.startsWith(href);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex flex-col">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-lg sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Logo */}
            <Link
              to="/"
              className="flex items-center gap-3 hover:opacity-80 transition-opacity"
              onClick={() => setMobileMenuOpen(false)}
            >
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                SwasthyaAI
              </span>
            </Link>

            {/* Desktop Navigation - Only for authenticated users */}
            {isAuthenticated && (
              <nav className="hidden lg:flex items-center gap-1">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = isActiveRoute(item.href);
                  return (
                    <Link key={item.name} to={item.href}>
                      <Button
                        variant={isActive ? "secondary" : "ghost"}
                        size="sm"
                        className={cn(
                          "gap-2 transition-all duration-200",
                          isActive
                            ? "shadow-sm bg-blue-50 text-blue-700 border-blue-200"
                            : "hover:bg-blue-50/50 hover:text-blue-600"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        {item.name}
                      </Button>
                    </Link>
                  );
                })}
              </nav>
            )}

            {/* Auth Buttons & Mobile Menu */}
            <div className="flex items-center gap-3">
              {isAuthenticated ? (
                <>
                  {/* Desktop Logout */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleLogout}
                    className="hidden sm:flex gap-2 hover:bg-red-50 hover:text-red-600"
                  >
                    <LogOut className="h-4 w-4" />
                    Logout
                  </Button>

                  {/* Mobile Menu Button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    className="lg:hidden p-2"
                  >
                    {mobileMenuOpen ? (
                      <X className="h-5 w-5" />
                    ) : (
                      <Menu className="h-5 w-5" />
                    )}
                  </Button>
                </>
              ) : (
                <div className="flex items-center gap-2">
                  <Link to="/login">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="hover:bg-blue-50 hover:text-blue-600"
                    >
                      Login
                    </Button>
                  </Link>
                  <Link to="/register">
                    <Button
                      variant="default"
                      size="sm"
                      className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-sm"
                    >
                      Sign Up
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Navigation Slide-in */}
        {isAuthenticated && mobileMenuOpen && (
          <div className="lg:hidden border-t bg-white/95 backdrop-blur-lg">
            <div className="container mx-auto px-4 py-4">
              <div className="flex flex-col gap-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = isActiveRoute(item.href);
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <Button
                        variant={isActive ? "secondary" : "ghost"}
                        size="lg"
                        className={cn(
                          "w-full justify-start gap-3 transition-all duration-200",
                          isActive
                            ? "shadow-sm bg-blue-50 text-blue-700 border-blue-200"
                            : "hover:bg-blue-50/50 hover:text-blue-600"
                        )}
                      >
                        <Icon className="h-5 w-5" />
                        {item.name}
                      </Button>
                    </Link>
                  );
                })}

                {/* Mobile Logout */}
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={handleLogout}
                  className="w-full justify-start gap-3 hover:bg-red-50 hover:text-red-600 text-red-600 mt-2"
                >
                  <LogOut className="h-5 w-5" />
                  Logout
                </Button>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t bg-white/50 backdrop-blur-sm mt-auto">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Brain className="h-4 w-4 text-white" />
              </div>
              <span className="font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                ML Predictor
              </span>
            </div>
            <p className="text-sm text-muted-foreground text-center sm:text-right">
              Developed by{" "}
              <span className="font-medium text-gray-700">Amir Pandit</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
