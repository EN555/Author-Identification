import { 
  Route,
  Redirect,
  Switch,
  BrowserRouter as Router
} from "react-router-dom";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import NotFound from './pages/notFound';
import ModelDashboardPage from "./pages/modelDashboardPage";
import InferencePage from "./pages/inferencePage";
import RetrainScreen from "./pages/retrainScreen";
import CustomNavbar from "./components/navbar";

function App() {
  console.log("starting app");
  return (
    <>
        <CustomNavbar />
        <main className="main">
          <ToastContainer />
          <Router>
            <Switch>
              <Route path="/retrain" component={RetrainScreen} />
              <Route path="/models" component={ModelDashboardPage}/>
              <Route path="/infer" component={InferencePage} />
              <Route
                path="/"
                exact="true"
                render={(props) => (
                  <Redirect
                    to={{
                      pathname: "/infer",
                      state: props.location,
                    }}
                  />)
                }
              ></Route>
              <Route path="/not-found" component={NotFound} />
              <Redirect to="/not-found" />
            </Switch>
          </Router>
        </main>
      </>
  );
}

export default App;
