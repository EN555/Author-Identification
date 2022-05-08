import React from 'react'
import { Navbar,Container,Nav } from 'react-bootstrap';

export default function CustomNavbar() {
  return (
    <div>
        <Navbar bg="dark" variant="dark">
            <Container>
            <Navbar.Brand href="/infer">Author Idenefication</Navbar.Brand>
            <Nav className="me-auto">
              <Nav.Link href="/infer">Inference</Nav.Link>
              <Nav.Link href="/retrain">Retrain</Nav.Link>
              <Nav.Link href="/models">Models</Nav.Link>
            </Nav>
            </Container>
          </Navbar>
    </div>
  )
}
