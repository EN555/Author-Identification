import { 
  Route,
  Redirect,
  Switch,
  BrowserRouter as Router
} from "react-router-dom";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import InferForm from './components/inferForm';
import NotFound from './components/notFound';

function App() {
  console.log("starting app");
  return (
    <>
        {/* <NavBar user={user} /> */}
        <main className="main">
          <ToastContainer />
          <Router>
            <Switch>
              <Route path="/infer" component={InferForm} />
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
