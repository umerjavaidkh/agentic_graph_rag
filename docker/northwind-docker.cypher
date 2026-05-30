// Northwind demo load for Docker (no Neo4j GenAI / no API keys).
// Requires APOC (enabled in docker-compose neo4j service).

CREATE CONSTRAINT Product_productID IF NOT EXISTS FOR (p:Product) REQUIRE (p.productID) IS UNIQUE;
CREATE CONSTRAINT Category_categoryID IF NOT EXISTS FOR (c:Category) REQUIRE (c.categoryID) IS UNIQUE;
CREATE CONSTRAINT Supplier_supplierID IF NOT EXISTS FOR (s:Supplier) REQUIRE (s.supplierID) IS UNIQUE;
CREATE CONSTRAINT Customer_customerID IF NOT EXISTS FOR (c:Customer) REQUIRE (c.customerID) IS UNIQUE;
CREATE CONSTRAINT Order_orderID IF NOT EXISTS FOR (o:Order) REQUIRE (o.orderID) IS UNIQUE;
CREATE CONSTRAINT Address_addressID IF NOT EXISTS FOR (a:Address) REQUIRE (a.addressID) IS UNIQUE;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/products.csv" AS row
MERGE (n:Product {productID:row.productID})
SET n += row,
n.unitPrice = toFloat(row.unitPrice),
n.unitsInStock = toInteger(row.unitsInStock), n.unitsOnOrder = toInteger(row.unitsOnOrder),
n.reorderLevel = toInteger(row.reorderLevel), n.discontinued = (row.discontinued <> "0");

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/categories.csv" AS row
MERGE (n:Category {categoryID:row.categoryID})
SET n += row;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/suppliers.csv" AS row
MERGE (n:Supplier {supplierID:row.supplierID})
SET n += row;

MATCH (p:Product),(c:Category)
WHERE p.categoryID = c.categoryID
MERGE (p)-[:BELONGS_TO]->(c);

MATCH (p:Product),(s:Supplier)
WHERE p.supplierID = s.supplierID
MERGE (s)<-[:SUPPLIED_BY]-(p);

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/customers.csv" AS row
MERGE (n:Customer {customerID:row.customerID})
SET n += row;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/orders.csv" AS row
MERGE (o:Order {orderID:row.orderID})
SET o.customerID = row.customerID,
    o.employeeID = row.employeeID,
    o.orderDate = CASE WHEN row.orderDate IS NOT NULL AND trim(row.orderDate) <> ''
        THEN date(substring(row.orderDate, 0, 10)) ELSE null END,
    o.requiredDate = row.requiredDate,
    o.shippedDate = row.shippedDate,
    o.shipVia = row.shipVia,
    o.freight = toFloat(row.freight)
MERGE (a:Address {addressID: apoc.text.join([coalesce(row.shipName, ''), coalesce(row.shipAddress, ''),
    coalesce(row.shipCity, ''), coalesce(row.shipRegion, ''), coalesce(row.shipPostalCode, ''),
    coalesce(row.shipCountry, '')], ', ')})
SET a.name = row.shipName,
    a.address = row.shipAddress,
    a.city = row.shipCity,
    a.region = row.shipRegion,
    a.postalCode = row.shipPostalCode,
    a.country = row.shipCountry
MERGE (o)-[:SHIPPED_TO]->(a)
WITH o
MATCH (c:Customer)
WHERE c.customerID = o.customerID
MERGE (c)-[:ORDERED]->(o);

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/order-details.csv" AS row
MATCH (p:Product), (o:Order)
WHERE p.productID = row.productID AND o.orderID = row.orderID
MERGE (o)-[details:ORDER_CONTAINS]->(p)
SET details = row,
details.unitPrice = toFloat(row.unitPrice),
details.quantity = toInteger(row.quantity),
details.discount = toFloat(row.discount);

MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
SET p.text = "Product Category: " + c.categoryName + " - " + coalesce(c.description, "")
  + "\nProduct Name: " + p.productName;
