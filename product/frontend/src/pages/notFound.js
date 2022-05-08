import React from "react";
import { Link } from "react-router-dom";

const NotFound = () => {
  console.log("in Notfound");
  return (
    <React.Fragment>
      <img alt="" src="images/404-back.jpg" height="500px" width="100%" style={{margin:"1rem"}}></img>
      <Link to="/" className="btn btn-primary">
        Go Home
      </Link>
    </React.Fragment>
  );
};

export default NotFound;
