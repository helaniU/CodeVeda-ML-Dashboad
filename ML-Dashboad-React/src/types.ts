declare module '*?raw' {
  const content: string;
  export default content;
}

export interface IrisData {
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  species: string;
}

export interface SentimentData {
  Text: string;
  Sentiment: string;
  Timestamp?: string;
  User?: string;
  Platform?: string;
  Hashtags?: string;
  Retweets?: number;
  Likes?: number;
  Country?: string;
}

export interface HouseData {
  CRIM: number;
  ZN: number;
  INDUS: number;
  CHAS: number;
  NOX: number;
  RM: number;
  AGE: number;
  DIS: number;
  RAD: number;
  TAX: number;
  PTRATIO: number;
  B: number;
  LSTAT: number;
  MEDV: number;
  [key: string]: number; // index signature for feature engineering
}
