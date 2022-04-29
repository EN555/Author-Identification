import React from "react";
import { Route, Redirect, Switch } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import InferForm from './components/inferForm';
import NotFound from './components/notFound';

function App() {
  console.log("starting app");
  return (
    <React.Fragment>
        {/* <NavBar user={user} /> */}
        <main className="container">
          <InferForm />
          {/* <ToastContainer />
          <Switch>
            <Route path="/infer" component={InferForm} />
            <Route
              path="/"
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
          </Switch> */}
        </main>
      </React.Fragment>
  );
}

export default App;
